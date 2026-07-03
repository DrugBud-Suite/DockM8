#!/usr/bin/env python3
"""Bounded-memory streaming pivot for DockM8 aggregation.

Replaces the canonical in-RAM ``long.pivot(...)`` -- which collects ~970M long rows
for one DEKOIS threshold into a 12.2M x 83 frame and OOMs with ``Cannot allocate
memory`` -- with a two-stage streaming build:

  1. Per-(target, threshold) SHARDS (:func:`build_target_shard`): one sorted row per
     workflow, carrying ALL requested metrics, filtered to a single threshold,
     written with a fixed row-group size so every shard for a threshold has
     identical, aligned physical chunks. A shard is reused for every metric at its
     threshold.
  2. Streaming WIDE output (:func:`stream_wide_parquet`): per metric, read row-group
     ``i`` from every target shard in lockstep, verify the workflow keys are
     identical, emit one output batch (anchor keys + one metric column per target,
     nulls -> 0.0) and write it straight through a ``pyarrow.parquet.ParquetWriter``.
     Only one aligned batch per target is ever resident; the full wide pivot is
     never materialized in Python.

Output is frame-identical to ``analysis.data_aggregation.pivot_metric_from_parquet``
on data whose targets share an identical workflow-key set (the real DEKOIS grid).
A differing key set fails loudly rather than silently dropping/inventing a row.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from analysis.data_aggregation import PIVOT_KEYS

# Bump when the shard/output encoding or alignment contract changes so persisted
# shards and final outputs from an older layout are invalidated, not trusted.
ALGO_VERSION = "stream-v1"

# Rows per shard row group == per streamed batch. All shards for a threshold are
# written with this size, so reading row group ``i`` from each yields aligned,
# bounded batches. ~256K rows x (4 str keys + 5 f64) per target is well under the
# 24 GiB ceiling even across ~83 targets.
DEFAULT_ROW_GROUP_ROWS = 262_144


# =============================================================================
# Grouping parts by their single target
# =============================================================================

def part_target(part_path: str | Path) -> str:
    """The single ``target`` value carried by a normalized part (read of one row)."""
    head = pl.read_parquet(str(part_path), columns=["target"], n_rows=1)
    if head.height == 0:
        raise ValueError(f"part has no rows; cannot determine target: {part_path}")
    return head["target"][0]


def group_parts_by_target(part_paths) -> dict[str, list[str]]:
    """Map each part to its single target -> {target: [part_path, ...]}.

    A part is one source performance.csv (one target, one docking, one selection),
    so a target owns several parts. Grouping is threshold-independent: do it once
    and reuse across thresholds.
    """
    groups: dict[str, list[str]] = {}
    for p in part_paths:
        groups.setdefault(part_target(p), []).append(str(p))
    return groups


# =============================================================================
# Target shards
# =============================================================================

def build_target_shard(
    parts, threshold, metrics: list[str], out_path: str | Path,
    *, row_group_size: int = DEFAULT_ROW_GROUP_ROWS,
) -> dict:
    """Build one atomic, row-group-chunked shard for a target at a threshold.

    Scans only this target's parts, keeps the workflow keys plus ALL requested
    metrics, filters to ``threshold``, fails loudly on a duplicate workflow key
    (the streaming analogue of polars ``aggregate_function=None``), sorts by the
    workflow keys, and writes with a fixed ``row_group_size`` so shards align.
    """
    parts = [str(p) for p in parts]
    df = (
        pl.scan_parquet(parts)
        .filter(pl.col("threshold") == float(threshold))
        .select([*PIVOT_KEYS, *metrics])
        .collect()
    )
    if df.select(PIVOT_KEYS).is_duplicated().any():
        raise ValueError(
            f"duplicate workflow key(s) in target shard at threshold {threshold} "
            f"-> {out_path}; each (docking, scoring, consensus_method, "
            f"selection_method) must be unique"
        )
    df = df.sort(PIVOT_KEYS)

    out_path = Path(out_path)
    tmp = f"{out_path}.tmp"
    df.write_parquet(tmp, row_group_size=row_group_size)
    os.replace(tmp, out_path)
    return {
        "algo_version": ALGO_VERSION,
        "threshold": threshold,
        "metrics": list(metrics),
        "rows": df.height,
        "row_groups": pq.ParquetFile(str(out_path)).num_row_groups,
    }


# =============================================================================
# Streaming wide output
# =============================================================================

def iter_wide_tables(shard_paths, targets: list[str], metric: str) -> Iterator[pa.Table]:
    """Yield aligned wide batches (anchor keys + one metric column per target).

    Reads row group ``i`` from every shard in lockstep. The first target anchors
    the workflow keys; every other shard's keys for that row group must be
    identical (same count AND same values) or aggregation fails -- it never
    silently drops or invents a workflow. Null metric values become 0.0.
    """
    shard_paths = [str(p) for p in shard_paths]
    if len(shard_paths) != len(targets):
        raise ValueError(
            f"shard/target length mismatch: {len(shard_paths)} shards, {len(targets)} targets"
        )

    pfs = [pq.ParquetFile(p) for p in shard_paths]
    nrg = pfs[0].num_row_groups
    for t, pf in zip(targets, pfs, strict=True):
        if pf.num_row_groups != nrg:
            raise ValueError(
                f"shard row-group count mismatch: target {t!r} has {pf.num_row_groups} "
                f"row groups != anchor {targets[0]!r} {nrg}"
            )

    anchor = pfs[0]
    anchor_schema = anchor.schema_arrow
    if metric not in anchor_schema.names:
        raise ValueError(f"metric {metric!r} not present in shard schema {anchor_schema.names}")

    metric_type = anchor_schema.field(metric).type
    out_schema = pa.schema(
        [anchor_schema.field(k) for k in PIVOT_KEYS]
        + [pa.field(t, metric_type) for t in targets]
    )

    for rg in range(nrg):
        anchor_keys = anchor.read_row_group(rg, columns=list(PIVOT_KEYS))
        n = anchor_keys.num_rows
        arrays = [anchor_keys.column(k) for k in PIVOT_KEYS]
        for t, pf in zip(targets, pfs, strict=True):
            tab = pf.read_row_group(rg, columns=[*PIVOT_KEYS, metric])
            if tab.num_rows != n:
                raise ValueError(
                    f"workflow key mismatch at target {t!r}, row group {rg}: "
                    f"{tab.num_rows} rows != anchor {targets[0]!r} {n}"
                )
            for k in PIVOT_KEYS:
                if not tab.column(k).equals(anchor_keys.column(k)):
                    raise ValueError(
                        f"workflow key mismatch at target {t!r}, row group {rg}, "
                        f"column {k!r}: shard keys differ from anchor {targets[0]!r}"
                    )
            arrays.append(pc.fill_null(tab.column(metric), 0.0))
        yield pa.Table.from_arrays(arrays, schema=out_schema)


def stream_wide_parquet(
    shard_paths, targets: list[str], metric: str, out_path: str | Path,
    *, row_group_size: int = DEFAULT_ROW_GROUP_ROWS,
) -> int:
    """Stream the wide pivot for one metric to an atomic, validated parquet.

    Writes bounded batches straight through a ``ParquetWriter`` to a temp file,
    re-reads it to confirm the row count, then atomically replaces the final
    output. Any failure (key mismatch, empty/absent threshold, truncated write)
    removes the temp file and never publishes a partial output.
    """
    out_path = Path(out_path)
    tmp = f"{out_path}.tmp"
    writer: pq.ParquetWriter | None = None
    written = 0
    try:
        for table in iter_wide_tables(shard_paths, targets, metric):
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table, row_group_size=row_group_size)
            written += table.num_rows
        if writer is None:
            raise ValueError(
                f"no rows produced for metric {metric!r}; threshold absent or shards empty"
            )
        writer.close()
        writer = None
        got = pq.ParquetFile(tmp).metadata.num_rows
        if got != written:
            raise ValueError(
                f"truncated output for {out_path}: wrote {written} rows, parquet has {got}"
            )
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    os.replace(tmp, out_path)
    return written


# =============================================================================
# Source signatures (resume binding)
# =============================================================================

def source_signature(part_paths) -> str:
    """Stable hash of a part set's identities: sorted (abspath, size, mtime_ns).

    Binds shards and final outputs to the exact source parts that produced them, so
    a regenerated/added/removed part invalidates the affected shards and outputs.
    """
    h = hashlib.sha1()
    h.update(ALGO_VERSION.encode())
    for p in sorted(os.path.abspath(str(x)) for x in part_paths):
        try:
            st = os.stat(p)
            stamp = f"{p}|{st.st_size}|{st.st_mtime_ns}"
        except OSError:
            stamp = f"{p}|missing"
        h.update(stamp.encode())
        h.update(b"\n")
    return h.hexdigest()


def is_valid_parquet(path: str | Path) -> bool:
    """True iff ``path`` is a readable parquet (guards against torn writes/kills)."""
    if not Path(path).exists():
        return False
    try:
        pl.read_parquet(str(path), n_rows=1)
        return True
    except Exception:  # noqa: BLE001
        return False


# =============================================================================
# Shard / output resume binding
# =============================================================================

def shard_reusable(entry: dict | None, signature: str, metrics: list[str],
                   shard_path: str | Path) -> bool:
    """Whether a persisted shard may be reused instead of rebuilt.

    Reusable only when its manifest entry exists, was written by THIS algorithm
    version, for the same metric SET (order is irrelevant -- a shard carries all
    requested metric columns by name), bound to the same source signature, and the
    shard file is still a readable parquet.
    """
    if entry is None:
        return False
    if entry.get("algo_version") != ALGO_VERSION:
        return False
    if sorted(entry.get("metrics", [])) != sorted(metrics):
        return False
    if entry.get("signature") != signature:
        return False
    return is_valid_parquet(shard_path)


OUTPUT_MANIFEST_NAME = "output_manifest.json"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return {}


def _write_json_atomic(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def read_output_manifest(out_dir: str | Path) -> dict:
    """Load the output manifest binding final outputs to a source signature."""
    return _read_json(Path(out_dir) / OUTPUT_MANIFEST_NAME)


def write_output_manifest(out_dir: str | Path, manifest: dict) -> None:
    """Persist the output manifest atomically."""
    _write_json_atomic(Path(out_dir) / OUTPUT_MANIFEST_NAME, manifest)


def record_outputs(out_dir: str | Path, produced: dict[str, dict], signature: str) -> None:
    """Bind newly produced final outputs to ``signature`` in the output manifest."""
    manifest = read_output_manifest(out_dir)
    manifest["algo_version"] = ALGO_VERSION
    outs = manifest.setdefault("outputs", {})
    for fname, meta in produced.items():
        outs[fname] = {"algo_version": ALGO_VERSION, "signature": signature, **meta}
    write_output_manifest(out_dir, manifest)


def invalidate_stale_outputs(out_dir: str | Path, expected_filenames: list[str],
                             signature: str) -> list[str]:
    """Remove final outputs not bound to ``signature`` via the output manifest.

    A final parquet is trusted (kept) only when its manifest entry exists, was
    written by this algorithm version, and matches the current source signature.
    Anything else among ``expected_filenames`` -- an unbound leftover, a
    stale-source binding, an older algorithm version -- is removed (with its
    sibling CSV) so the run rebuilds and re-binds it. Returns the basenames removed.
    """
    out_dir = Path(out_dir)
    manifest = read_output_manifest(out_dir)
    entries = manifest.setdefault("outputs", {})
    removed: list[str] = []
    for fname in expected_filenames:
        pq_path = out_dir / fname
        entry = entries.get(fname)
        bound = (
            entry is not None
            and entry.get("algo_version") == ALGO_VERSION
            and entry.get("signature") == signature
        )
        if bound:
            continue
        entries.pop(fname, None)
        if pq_path.exists():
            pq_path.unlink()
            csv_path = pq_path.with_suffix(".csv")
            if csv_path.exists():
                csv_path.unlink()
            removed.append(fname)
    manifest["outputs"] = entries
    write_output_manifest(out_dir, manifest)
    return removed
