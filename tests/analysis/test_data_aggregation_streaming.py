# Parity: the shipped streaming aggregation (run_aggregation_pipeline, which uses the
# bounded-memory streaming pivot) is frame-identical to the canonical in-RAM
# pivot_metric_from_parquet, on data whose targets share a workflow-key grid.
import polars as pl

from analysis.data_aggregation import (
    consolidate_to_parquet,
    find_all_performance_files,
    pivot_metric_from_parquet,
    run_aggregation_pipeline,
)

PIVOT_KEYS = ["docking", "scoring", "consensus_method", "selection_method"]


def _make_dataset(base, targets_ef):
    for tgt, ef0 in targets_ef.items():
        d = base / tgt / "results" / "performance"
        d.mkdir(parents=True)
        pl.DataFrame({
            "scoring_function": ["AD4", "CNN-Score", None],
            "combination": [None, None, "CNN-Score+AD4"],  # unsorted -> canonicalized on read
            "consensus_method": [None, None, "ecr"],
            "threshold": [1.0, 1.0, 1.0],
            "ef": [ef0, ef0 + 2.0, ef0 + 4.0],
            "auc_roc": [0.80, 0.90, 0.95],
        }).write_csv(d / f"{tgt}_gnina_bestpose_performance.csv")


def test_streaming_matches_canonical_pivot(tmp_path):
    base = tmp_path / "data"
    out = tmp_path / "out"
    _make_dataset(base, {"aaa": 5.0, "bbb": 6.0, "ccc": 7.0})

    run_aggregation_pipeline(
        str(base), str(out), metrics=["ef", "auc_roc"],
        thresholds=[1], skip_existing=False, verbose=False,
    )
    streamed = pl.read_parquet(out / "combined_results_ef_pivot_thresh1.parquet")

    # Canonical in-RAM oracle over the same consolidated parts.
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    consolidate_to_parquet(find_all_performance_files(str(base)), str(scratch), ["ef", "auc_roc"])
    canonical = pivot_metric_from_parquet(str(scratch / "*.parquet"), "ef", 1)

    assert streamed.sort(PIVOT_KEYS).equals(canonical.sort(PIVOT_KEYS))
    assert set(streamed.columns) == set(PIVOT_KEYS + ["aaa", "bbb", "ccc"])
    # both CSV and Parquet are written, and the threshold-invariant metric too
    assert (out / "combined_results_ef_pivot_thresh1.csv").exists()
    assert (out / "combined_results_auc_roc_pivot_thresh1.parquet").exists()
