#!/usr/bin/env python3
"""
Data Aggregation Script for DockM8 Performance Results.

This script aggregates performance data from individual target folders into
pivot tables with methods as rows and targets as columns.

Supports multiple datasets: DEKOIS, DUD-E, Lit-PCBA

Output files are saved in both CSV and Parquet formats.

Usage:
    python data_aggregation.py --dataset DEKOIS --base-dir /path/to/data
    python data_aggregation.py --dataset DUD-E --skip-existing
    python data_aggregation.py --dataset Lit-PCBA --metrics ref,ef --thresholds 1,5
"""

import argparse
import os
import shutil
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import polars as pl
from tqdm import tqdm

from .config import get_base_path, get_aggregated_dir, DATASETS

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_METRICS = ["auc_roc", "bedroc", "pm", "ef", "ref"]
DEFAULT_THRESHOLDS = [0.1, 0.5, 1, 2, 5, 10]
FILE_PREFIX = "combined_results"

DATASET_SUBDIRS = {
    "DEKOIS": "DEKOIS_2.0x",
    "DUD-E": "DUD-E",
    "Lit-PCBA": "lit-pcba",
}

DATASET_OUTPUT_SUBDIRS = {
    "DEKOIS": "DEKOIS_aggregated_results",
    "DUD-E": "DUD-E_aggregated_results",
    "Lit-PCBA": "lit-pcba_aggregated_results",
}


def get_dataset_dirs(dataset: str, base_path: Path | None = None) -> tuple[Path, Path]:
    """
    Get input and output directories for a dataset.

    Args:
        dataset: Dataset name (DEKOIS, DUD-E, or Lit-PCBA)
        base_path: Base path (uses config default if None)

    Returns:
        Tuple of (base_dir, output_dir)
    """
    if base_path is None:
        base_path = get_base_path()

    base_dir = base_path / DATASET_SUBDIRS.get(dataset, dataset)
    output_dir = get_aggregated_dir() / DATASET_OUTPUT_SUBDIRS.get(dataset, f"{dataset}_aggregated_results")

    return base_dir, output_dir


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_metadata_from_filename(filename: str) -> tuple[str, str, str]:
    """
    Extract target, docking and selection method from filename.

    Filename pattern: {target}_{docking}_{selection}_performance.csv

    Args:
        filename: Path to the performance CSV file

    Returns:
        Tuple of (target, docking_method, selection_method)
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')

    target = parts[0]
    docking = parts[1]
    selection = parts[2]

    return target, docking, selection


def find_all_performance_files(base_dir: str, targets: list[str] | None = None) -> list[str]:
    """
    Find all performance files across all targets.

    Args:
        base_dir: Base directory containing target folders
        targets: Optional list of specific targets to process

    Returns:
        List of file paths to performance CSV files
    """
    # Find every results/performance dir at ANY depth, so this works for both the
    # flat dataset/<target>/ layout and the nested dataset/PART_N/<target>/ layout
    # (Lit-PCBA on the SSD mirror). An immediate-listdir would silently skip PART_N.
    target_filter = set(targets) if targets else None
    all_files: list[str] = []
    for perf_dir in tqdm(sorted(Path(base_dir).rglob("performance")), desc="Finding files"):
        if perf_dir.parent.name != "results":
            continue
        for path in perf_dir.glob("*_performance.csv"):
            if target_filter is None or path.name.split("_")[0] in target_filter:
                all_files.append(str(path))

    return all_files


# Pivot index: methods are rows, targets become columns.
PIVOT_KEYS = ["docking", "scoring", "consensus_method", "selection_method"]


def normalize_performance_file(file_path: str, metrics: list[str]) -> pl.DataFrame | None:
    """
    Read one performance CSV ONCE into long format.

    Retains every requested metric and all thresholds.
    Canonicalizes the consensus ``combination`` (order-insensitive), derives the
    ``scoring`` column, and tags target/docking/selection from the filename --
    the same transforms the old per-(metric, threshold) path applied on every
    pass, now done a single time per file so the source CSV is read only once.

    Returns None if the file is missing/empty or has none of the requested metrics.
    """
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return None

    target_name, docking_method, selection_method = extract_metadata_from_filename(file_path)

    try:
        df = pl.read_csv(file_path, ignore_errors=True)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    available = set(df.columns)
    if "threshold" not in available or not any(m in available for m in metrics):
        return None

    for col in ("scoring_function", "combination", "consensus_method"):
        if col not in available:
            df = df.with_columns(pl.lit(None).alias(col))
    for metric in metrics:
        if metric not in available:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(metric))

    df = df.with_columns(
        pl.when(pl.col("combination").is_null())
        .then(pl.col("combination"))
        .otherwise(
            pl.col("combination").str.split("+").list.eval(pl.element().sort()).list.join("+")
        )
        .alias("combination")
    )
    df = df.with_columns(
        pl.when(pl.col("scoring_function").is_null() | (pl.col("scoring_function") == ""))
        .then(pl.col("combination"))
        .otherwise(pl.col("scoring_function"))
        .alias("scoring"),
        pl.col("consensus_method").fill_null(""),
    )

    return df.select(
        pl.col("threshold").cast(pl.Float64),
        pl.lit(docking_method).alias("docking"),
        pl.col("scoring"),
        pl.col("consensus_method"),
        pl.lit(selection_method).alias("selection_method"),
        pl.lit(target_name).alias("target"),
        *[pl.col(m).cast(pl.Float64) for m in metrics],
    )


def consolidate_to_parquet(files: list[str], scratch_dir: str, metrics: list[str]) -> int:
    """
    Read every performance CSV once and write a compact long-format parquet per file.

    Writes into ``scratch_dir`` and returns the number of files that yielded data.
    Bounded memory: only one CSV is held at a time, so this scales to the
    billions of rows in the largest datasets (DEKOIS) without an in-RAM concat.
    """
    written = 0
    for i, file_path in enumerate(tqdm(files, desc="Consolidating files")):
        df = normalize_performance_file(file_path, metrics)
        if df is None:
            continue
        df.write_parquet(os.path.join(scratch_dir, f"part_{i:06d}.parquet"))
        written += 1
    return written


def pivot_metric_from_parquet(scratch_glob: str, metric_name: str, threshold: float) -> pl.DataFrame:
    """
    Build one pivot table (methods x targets) for a metric/threshold.

    Scans the consolidated parquet with predicate + projection pushdown, then does
    a single native ``pivot``.
    ``aggregate_function=None`` is intentional: each (method, target) cell must
    hold exactly one value. A duplicate raises loudly rather than silently
    picking one -- consistent with the recompute pipeline's fail-loud policy.
    """
    long = (
        pl.scan_parquet(scratch_glob)
        .filter(pl.col("threshold") == float(threshold))
        .select([*PIVOT_KEYS, "target", metric_name])
        .collect()
    )
    pivot = long.pivot(values=metric_name, index=PIVOT_KEYS, on="target", aggregate_function=None)
    target_cols = sorted(c for c in pivot.columns if c not in PIVOT_KEYS)
    return pivot.select([*PIVOT_KEYS, *target_cols]).fill_null(0).sort(PIVOT_KEYS)


def save_pivot_table(
    df: pl.DataFrame,
    output_dir: str,
    file_prefix: str,
    metric_name: str,
    threshold: float
) -> tuple[str, str]:
    """
    Save the pivot table to both CSV and Parquet formats.

    Args:
        df: Pivot table DataFrame
        output_dir: Output directory path
        file_prefix: Prefix for output filename
        metric_name: Name of the metric
        threshold: Threshold value

    Returns:
        Tuple of (csv_filepath, parquet_filepath)
    """
    os.makedirs(output_dir, exist_ok=True)

    threshold_str = str(threshold).replace('.', 'p')

    # Write to a temp file then atomically rename, so an interrupted/failed write
    # never leaves a half-written pivot in place of a good one.
    csv_filename = f"{file_prefix}_{metric_name}_pivot_thresh{threshold_str}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    df.write_csv(csv_filepath + ".tmp")
    os.replace(csv_filepath + ".tmp", csv_filepath)
    print(f"CSV saved to: {csv_filepath}")

    parquet_filename = f"{file_prefix}_{metric_name}_pivot_thresh{threshold_str}.parquet"
    parquet_filepath = os.path.join(output_dir, parquet_filename)
    df.write_parquet(parquet_filepath + ".tmp")
    os.replace(parquet_filepath + ".tmp", parquet_filepath)
    print(f"Parquet saved to: {parquet_filepath}")

    return csv_filepath, parquet_filepath


def check_output_exists(
    output_dir: str,
    file_prefix: str,
    metric_name: str,
    threshold: float
) -> bool:
    """
    Check if output files already exist.

    Args:
        output_dir: Output directory path
        file_prefix: Prefix for output filename
        metric_name: Name of the metric
        threshold: Threshold value

    Returns:
        True if both CSV and Parquet files exist
    """
    threshold_str = str(threshold).replace('.', 'p')
    csv_filename = f"{file_prefix}_{metric_name}_pivot_thresh{threshold_str}.csv"
    parquet_filename = f"{file_prefix}_{metric_name}_pivot_thresh{threshold_str}.parquet"

    csv_path = os.path.join(output_dir, csv_filename)
    parquet_path = os.path.join(output_dir, parquet_filename)

    return os.path.exists(csv_path) and os.path.exists(parquet_path)


def _stream_csv_from_parquet(parquet_path: str, csv_path: str) -> None:
    """Stream a wide-pivot parquet to CSV without loading it fully into RAM."""
    tmp = csv_path + ".tmp"
    pl.scan_parquet(parquet_path).sink_csv(tmp)
    os.replace(tmp, csv_path)


def _run_streaming_pivots(
    scratch_dir: str,
    output_dir: str,
    file_prefix: str,
    pending: list[tuple[str, float]],
    metrics: list[str],
    verbose: bool = True,
) -> list[tuple[str, float, str]]:
    """Build each pending (metric, threshold) pivot with bounded memory.

    Replaces the in-RAM ``pivot_metric_from_parquet`` (which collects ~970M long rows
    for one DEKOIS threshold and OOMs). For each threshold, builds one aligned shard
    per target carrying all metrics (:func:`analysis.streaming_pivot.build_target_shard`),
    then streams the wide pivot per metric straight to an atomic parquet
    (:func:`analysis.streaming_pivot.stream_wide_parquet`) and a streamed CSV -- the full
    12.2M x 83 frame is never materialized. Output is frame-identical to
    ``pivot_metric_from_parquet`` on the real grid (targets share a workflow-key set);
    a differing key set fails loudly rather than silently 0-filling.

    Returns a list of (metric, threshold, error) for any failures.
    """
    from collections import defaultdict

    from .streaming_pivot import build_target_shard, group_parts_by_target, stream_wide_parquet

    parts = sorted(str(p) for p in Path(scratch_dir).glob("*.parquet"))
    by_target = group_parts_by_target(parts)
    targets = sorted(by_target)
    shard_dir = Path(scratch_dir) / "shards"
    shard_dir.mkdir(exist_ok=True)

    by_threshold: dict[float, list[str]] = defaultdict(list)
    for metric, threshold in pending:
        by_threshold[threshold].append(metric)

    failures: list[tuple[str, float, str]] = []
    for threshold, metrics_at_threshold in by_threshold.items():
        threshold_str = str(threshold).replace(".", "p")
        try:
            shard_paths: list[str] = []
            for target in targets:
                shard = shard_dir / f"shard_thresh{threshold_str}_{target}.parquet"
                build_target_shard(by_target[target], threshold, metrics, shard)
                shard_paths.append(str(shard))
        except Exception as e:  # a bad shard set invalidates every metric at this threshold
            print(f"Error building shards at threshold {threshold}: {e}")
            print(traceback.format_exc())
            failures.extend((metric, threshold, str(e)) for metric in metrics_at_threshold)
            continue

        for metric in metrics_at_threshold:
            start_time = time.time()
            base = f"{file_prefix}_{metric}_pivot_thresh{threshold_str}"
            out_parquet = os.path.join(output_dir, base + ".parquet")
            try:
                rows = stream_wide_parquet(shard_paths, targets, metric, out_parquet)
                _stream_csv_from_parquet(out_parquet, os.path.join(output_dir, base + ".csv"))
                if verbose:
                    print(f"  {metric} @ {threshold}: {rows} rows x {len(targets)} targets "
                          f"-> {base}.{{parquet,csv}} ({time.time() - start_time:.2f}s)")
            except Exception as e:
                print(f"Error processing metric {metric} at threshold {threshold}: {e}")
                print(traceback.format_exc())
                failures.append((metric, threshold, str(e)))

        for shard_path in shard_paths:  # free this threshold's shards before the next
            Path(shard_path).unlink(missing_ok=True)

    return failures


def run_aggregation_pipeline(
    base_dir: str,
    output_dir: str,
    file_prefix: str = FILE_PREFIX,
    metrics: list[str] | None = None,
    thresholds: list[float] | None = None,
    skip_existing: bool = True,
    verbose: bool = True
) -> None:
    """
    Run the data aggregation pipeline for all metrics and thresholds.

    Args:
        base_dir: Base directory containing target folders
        output_dir: Directory to save results
        file_prefix: Prefix for output files
        metrics: List of metrics to process (default: all)
        thresholds: List of thresholds to process (default: all)
        skip_existing: If True, skip processing for existing output files
        verbose: If True, print detailed progress information
    """
    total_start_time = time.time()

    if metrics is None:
        metrics = DEFAULT_METRICS
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    os.makedirs(output_dir, exist_ok=True)

    # Build the (metric, threshold) worklist; auc_roc/bedroc are threshold-invariant.
    # Keep each threshold's original value (int vs float) so output filenames match
    # the canonical naming (e.g. thresh1, not thresh1p0).
    worklist: list[tuple[str, float]] = []
    for metric in metrics:
        ths = [thresholds[0]] if metric in ("auc_roc", "bedroc") else thresholds
        for threshold in ths:
            worklist.append((metric, threshold))

    pending = [
        (metric, threshold)
        for metric, threshold in worklist
        if not (skip_existing and check_output_exists(output_dir, file_prefix, metric, threshold))
    ]
    for metric, threshold in worklist:
        if (metric, threshold) not in pending:
            print(f"Skipping metric={metric}, threshold={threshold} - output files already exist")
    if not pending:
        print("All aggregated outputs already exist; nothing to do.")
        return

    all_files = find_all_performance_files(base_dir)
    if not all_files:
        raise ValueError("No performance files found.")
    print(f"Found {len(all_files)} performance files.")

    # READ ONCE: consolidate every CSV into a compact parquet scratch set (all
    # requested metrics + all thresholds retained), then derive each pivot from
    # it -- instead of re-reading every file once per (metric, threshold).
    scratch_dir = tempfile.mkdtemp(prefix="dockm8_agg_")
    failures: list[tuple[str, float, str]] = []
    try:
        consolidate_start = time.time()
        n_loaded = consolidate_to_parquet(all_files, scratch_dir, list(metrics))
        if n_loaded == 0:
            raise ValueError("No data was loaded. Check file paths and formats.")
        print(f"Consolidated {n_loaded}/{len(all_files)} files in "
              f"{time.time() - consolidate_start:.2f} seconds")

        # Bounded-memory streaming pivot: never materialize the full 12.2M x 83 wide
        # frame in RAM (the in-RAM pivot_metric_from_parquet OOMs on full DEKOIS).
        failures = _run_streaming_pivots(
            scratch_dir, output_dir, file_prefix, pending, list(metrics), verbose=verbose
        )
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    total_end_time = time.time()
    print(f"\nTotal pipeline execution time: {total_end_time - total_start_time:.2f} seconds")
    # Do not report success if any (metric, threshold) failed -- raise so the caller
    # (and any resume/manifest logic) sees the failure instead of a silent partial run.
    if failures:
        summary = ", ".join(f"{m}@{t}" for m, t, _ in failures)
        raise RuntimeError(f"aggregation failed for {len(failures)} metric/threshold combos: {summary}")
    print(f"Results saved to: {output_dir}")


def aggregate_dataset_results(base_path: Path, dataset: str) -> None:
    """
    Aggregate performance CSVs for a single dataset into parquet pivot tables.

    Reads from: base_path/{dataset_subdir}/{target}/results/performance/
    Writes to:  results/output/aggregated/{dataset}_aggregated_results/
    """
    base_dir, output_dir = get_dataset_dirs(dataset, base_path)
    print(f"  Input:  {base_dir}")
    print(f"  Output: {output_dir}")
    run_aggregation_pipeline(str(base_dir), str(output_dir))


def parse_list_arg(arg_value: str | None, default: list) -> list:
    """Parse comma-separated argument into list."""
    if arg_value is None:
        return default
    return [x.strip() for x in arg_value.split(',')]


def parse_threshold_arg(arg_value: str | None) -> list[float] | None:
    """Parse comma-separated threshold argument into list of floats."""
    if arg_value is None:
        return None
    return [float(x.strip()) for x in arg_value.split(',')]


def main():
    """CLI entry point for standalone dataset aggregation."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=DATASETS,
        help='Dataset to process (DEKOIS, DUD-E, or Lit-PCBA)'
    )

    parser.add_argument(
        '--base-path',
        type=Path,
        default=None,
        help='Base path to raw benchmark data (Zenodo download)'
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Direct base directory containing target folders (overrides --base-path)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: auto-generated from base-path)'
    )

    parser.add_argument(
        '--metrics',
        type=str,
        default=None,
        help=f'Comma-separated list of metrics (default: {",".join(DEFAULT_METRICS)})'
    )

    parser.add_argument(
        '--thresholds',
        type=str,
        default=None,
        help=f'Comma-separated list of thresholds (default: {",".join(map(str, DEFAULT_THRESHOLDS))})'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip processing if output files already exist (default: True)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reprocessing even if output files exist'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    if args.base_dir:
        base_dir = args.base_dir
        output_dir = args.output_dir if args.output_dir else str(
            Path(base_dir).parent / DATASET_OUTPUT_SUBDIRS.get(args.dataset, f"{args.dataset}_aggregated_results")
        )
    else:
        base_path = get_base_path(args.base_path)
        base_dir_path, output_dir_path = get_dataset_dirs(args.dataset, base_path)
        base_dir = str(base_dir_path)
        output_dir = args.output_dir if args.output_dir else str(output_dir_path)

    metrics = parse_list_arg(args.metrics, DEFAULT_METRICS) if args.metrics else None
    thresholds = parse_threshold_arg(args.thresholds)

    skip_existing = not args.force

    print(f"Dataset: {args.dataset}")
    print(f"Base Directory: {base_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Metrics: {metrics if metrics else DEFAULT_METRICS}")
    print(f"Thresholds: {thresholds if thresholds else DEFAULT_THRESHOLDS}")
    print(f"Skip Existing: {skip_existing}")

    if not os.path.exists(base_dir):
        print(f"Error: Base directory does not exist: {base_dir}")
        return 1

    run_aggregation_pipeline(
        base_dir=base_dir,
        output_dir=output_dir,
        metrics=metrics,
        thresholds=thresholds,
        skip_existing=skip_existing,
        verbose=not args.quiet
    )

    return 0


if __name__ == "__main__":
    exit(main())
