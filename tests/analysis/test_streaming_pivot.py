"""Tests for the bounded-memory streaming pivot primitives.

These pin the streaming-aggregation design contract:
  * a per-target shard carries one sorted row per workflow and ALL requested
    metrics, filtered to a single threshold, with a fail-loud duplicate check;
  * the wide writer consumes BOUNDED record batches (one per row group) instead
    of a complete in-memory pivot frame, so peak memory is one batch per target;
  * the streamed wide parquet is frame-identical to the canonical polars pivot
    (same rows, columns, order, fill_null(0.0), dtype) on representative data;
  * mismatched target key sets fail loudly (never silently drop/invent a row);
  * outputs are written atomically (temp -> validate -> os.replace);
  * a source signature changes when any source part is rewritten.
"""

import sys
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest
from polars.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.data_aggregation import PIVOT_KEYS, pivot_metric_from_parquet
from analysis.streaming_pivot import (
    ALGO_VERSION,
    build_target_shard,
    group_parts_by_target,
    invalidate_stale_outputs,
    is_valid_parquet,
    iter_wide_tables,
    record_outputs,
    shard_reusable,
    source_signature,
    stream_wide_parquet,
)

METRICS = ["auc_roc", "bedroc"]


def _part(d: Path, target: str, docking: str, selection: str,
          rows: list[dict], metrics=METRICS) -> str:
    """Write one normalized long-format part (normalize_performance_file's schema)."""
    n = len(rows)
    data = {
        "threshold": pl.Series([r["threshold"] for r in rows], dtype=pl.Float64),
        "docking": pl.Series([docking] * n, dtype=pl.Utf8),
        "scoring": pl.Series([r["scoring"] for r in rows], dtype=pl.Utf8),
        "consensus_method": pl.Series([r.get("consensus_method", "") for r in rows], dtype=pl.Utf8),
        "selection_method": pl.Series([selection] * n, dtype=pl.Utf8),
        "target": pl.Series([target] * n, dtype=pl.Utf8),
    }
    for m in metrics:
        data[m] = pl.Series([r.get(m) for r in rows], dtype=pl.Float64)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{target}__{docking}__{selection}.parquet"
    pl.DataFrame(data).write_parquet(p)
    return str(p)


# --------------------------------------------------------------------------- #
# Grouping parts by their single target
# --------------------------------------------------------------------------- #

def test_group_parts_by_target(tmp_path):
    p1 = _part(tmp_path, "t1", "gnina", "best", [{"threshold": 0.1, "scoring": "A"}])
    p2 = _part(tmp_path, "t1", "smina", "best", [{"threshold": 0.1, "scoring": "A"}])
    p3 = _part(tmp_path, "t2", "gnina", "best", [{"threshold": 0.1, "scoring": "A"}])
    g = group_parts_by_target([p1, p2, p3])
    assert set(g) == {"t1", "t2"}
    assert sorted(g["t1"]) == sorted([p1, p2])
    assert g["t2"] == [p3]


# --------------------------------------------------------------------------- #
# Target shard build: sort, threshold filter, all-metrics, bounded row groups
# --------------------------------------------------------------------------- #

def test_build_target_shard_sorts_filters_and_chunks(tmp_path):
    parts = [_part(tmp_path / "p", "t1", "gnina", "best", [
        {"threshold": 0.1, "scoring": "B", "auc_roc": 0.2, "bedroc": 0.3},
        {"threshold": 0.1, "scoring": "A", "auc_roc": 0.4, "bedroc": 0.5},
        {"threshold": 0.5, "scoring": "A", "auc_roc": 0.9, "bedroc": 0.9},  # dropped
    ])]
    out = tmp_path / "shard.parquet"
    info = build_target_shard(parts, 0.1, METRICS, out, row_group_size=1)
    df = pl.read_parquet(out)
    assert df.columns == [*PIVOT_KEYS, *METRICS]   # workflow keys + ALL metrics
    assert df["scoring"].to_list() == ["A", "B"]   # sorted by workflow keys
    assert df.height == 2                           # other threshold filtered out
    assert info["rows"] == 2
    assert pq.ParquetFile(out).num_row_groups == 2  # row_group_size=1 -> bounded chunks
    assert not Path(str(out) + ".tmp").exists()     # atomic publish


def test_build_target_shard_raises_on_duplicate_workflow(tmp_path):
    parts = [
        _part(tmp_path / "a", "t1", "gnina", "best",
              [{"threshold": 0.1, "scoring": "A", "auc_roc": 0.1, "bedroc": 0.1}]),
        _part(tmp_path / "b", "t1", "gnina", "best",
              [{"threshold": 0.1, "scoring": "A", "auc_roc": 0.2, "bedroc": 0.2}]),
    ]
    with pytest.raises(ValueError):
        build_target_shard(parts, 0.1, METRICS, tmp_path / "s.parquet", row_group_size=8)


# --------------------------------------------------------------------------- #
# Bounded wide batches: the writer never sees a complete pivot frame
# --------------------------------------------------------------------------- #

def test_iter_wide_tables_are_bounded(tmp_path):
    def rows():
        return [{"threshold": 0.1, "scoring": s, "auc_roc": 0.1, "bedroc": 0.2}
                for s in ["A", "B", "C"]]
    s1 = tmp_path / "s1.parquet"
    s2 = tmp_path / "s2.parquet"
    build_target_shard([_part(tmp_path / "p1", "t1", "g", "b", rows())], 0.1, METRICS, s1, row_group_size=2)
    build_target_shard([_part(tmp_path / "p2", "t2", "g", "b", rows())], 0.1, METRICS, s2, row_group_size=2)
    tables = list(iter_wide_tables([s1, s2], ["t1", "t2"], "auc_roc"))
    assert len(tables) >= 2                          # streamed in >1 bounded unit
    assert all(t.num_rows <= 2 for t in tables)      # each <= row_group_size
    assert sum(t.num_rows for t in tables) == 3
    assert tables[0].column_names == [*PIVOT_KEYS, "t1", "t2"]


# --------------------------------------------------------------------------- #
# Parity with the canonical in-memory polars pivot
# --------------------------------------------------------------------------- #

def test_stream_wide_parquet_matches_canonical(tmp_path):
    def mk(target):
        return _part(tmp_path / "parts", target, "gnina", "best", [
            {"threshold": 0.1, "scoring": "A", "auc_roc": 0.7, "bedroc": 0.1},
            {"threshold": 0.1, "scoring": "B", "auc_roc": None, "bedroc": 0.2},  # null -> 0.0
        ])
    p_t1, p_t2 = mk("t1"), mk("t2")
    canon = pivot_metric_from_parquet([p_t1, p_t2], "auc_roc", 0.1)

    s1 = tmp_path / "s1.parquet"
    s2 = tmp_path / "s2.parquet"
    build_target_shard([p_t1], 0.1, METRICS, s1, row_group_size=8)
    build_target_shard([p_t2], 0.1, METRICS, s2, row_group_size=8)
    out = tmp_path / "out.parquet"
    stream_wide_parquet([s1, s2], ["t1", "t2"], "auc_roc", out, row_group_size=8)

    assert_frame_equal(pl.read_parquet(out), canon)
    assert not Path(str(out) + ".tmp").exists()


def test_stream_wide_parquet_unequal_keys_fails_without_output(tmp_path):
    s1 = tmp_path / "s1.parquet"
    s2 = tmp_path / "s2.parquet"
    build_target_shard([_part(tmp_path / "p1", "t1", "g", "b", [
        {"threshold": 0.1, "scoring": "A", "auc_roc": 0.1, "bedroc": 0.1},
        {"threshold": 0.1, "scoring": "B", "auc_roc": 0.1, "bedroc": 0.1}])],
        0.1, METRICS, s1, row_group_size=8)
    build_target_shard([_part(tmp_path / "p2", "t2", "g", "b", [
        {"threshold": 0.1, "scoring": "A", "auc_roc": 0.1, "bedroc": 0.1}])],  # missing B
        0.1, METRICS, s2, row_group_size=8)
    out = tmp_path / "o.parquet"
    with pytest.raises(ValueError):
        stream_wide_parquet([s1, s2], ["t1", "t2"], "auc_roc", out, row_group_size=8)
    assert not out.exists()
    assert not Path(str(out) + ".tmp").exists()


# --------------------------------------------------------------------------- #
# Source signature for resume binding
# --------------------------------------------------------------------------- #

def test_source_signature_changes_on_part_rewrite(tmp_path):
    p = _part(tmp_path, "t1", "g", "b",
              [{"threshold": 0.1, "scoring": "A", "auc_roc": 0.1, "bedroc": 0.1}])
    s1 = source_signature([p])
    _part(tmp_path, "t1", "g", "b", [  # rewrite same path, larger -> size+mtime change
        {"threshold": 0.1, "scoring": "A", "auc_roc": 0.1, "bedroc": 0.1},
        {"threshold": 0.1, "scoring": "B", "auc_roc": 0.2, "bedroc": 0.2}])
    assert source_signature([p]) != s1


# --------------------------------------------------------------------------- #
# Shard reuse binding (algo version + metric set + source signature + readable)
# --------------------------------------------------------------------------- #

def test_is_valid_parquet(tmp_path):
    good = tmp_path / "g.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(good)
    bad = tmp_path / "b.parquet"
    bad.write_bytes(b"not-a-parquet")
    assert is_valid_parquet(good) is True
    assert is_valid_parquet(bad) is False
    assert is_valid_parquet(tmp_path / "missing.parquet") is False


def test_shard_reusable_only_when_bound_and_readable(tmp_path):
    p = _part(tmp_path / "p", "t1", "g", "b",
              [{"threshold": 0.1, "scoring": "A", "auc_roc": 0.1, "bedroc": 0.1}])
    shard = tmp_path / "s.parquet"
    build_target_shard([p], 0.1, METRICS, shard, row_group_size=8)
    sig = source_signature([p])
    entry = {"algo_version": ALGO_VERSION, "metrics": METRICS, "signature": sig, "threshold": 0.1}

    assert shard_reusable(entry, sig, METRICS, shard) is True
    assert shard_reusable(entry, sig, list(reversed(METRICS)), shard) is True  # reorder ok
    assert shard_reusable(entry, "OTHER-SIG", METRICS, shard) is False         # source changed
    assert shard_reusable(entry, sig, ["auc_roc"], shard) is False             # metric set changed
    assert shard_reusable({**entry, "algo_version": "old"}, sig, METRICS, shard) is False
    assert shard_reusable(None, sig, METRICS, shard) is False
    shard.write_bytes(b"corrupt")
    assert shard_reusable(entry, sig, METRICS, shard) is False                 # truncated/corrupt


# --------------------------------------------------------------------------- #
# Output manifest: final parquets trusted only when bound to current sources
# --------------------------------------------------------------------------- #

def test_invalidate_stale_outputs_keeps_bound_removes_changed(tmp_path):
    name = "combined_results_auc_roc_pivot_thresh0p1.parquet"
    (tmp_path / name).write_bytes(b"x")
    record_outputs(tmp_path, {name: {"metric": "auc_roc", "threshold": 0.1, "rows": 1}}, "SIG1")

    assert invalidate_stale_outputs(tmp_path, [name], "SIG1") == []         # unchanged -> kept
    assert (tmp_path / name).exists()

    assert invalidate_stale_outputs(tmp_path, [name], "SIG2") == [name]     # source changed -> removed
    assert not (tmp_path / name).exists()


def test_invalidate_removes_unbound_existing_output_and_sibling_csv(tmp_path):
    pq_name = "combined_results_ef_pivot_thresh1.parquet"
    csv_name = "combined_results_ef_pivot_thresh1.csv"
    (tmp_path / pq_name).write_bytes(b"x")   # exists with NO manifest binding
    (tmp_path / csv_name).write_bytes(b"x")
    assert invalidate_stale_outputs(tmp_path, [pq_name], "SIG") == [pq_name]
    assert not (tmp_path / pq_name).exists()
    assert not (tmp_path / csv_name).exists()
