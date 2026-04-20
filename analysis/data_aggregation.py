#!/usr/bin/env python3
"""
Data Aggregation Script for DockM8 Performance Results

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
import glob
import os
import time
import traceback
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
from tqdm import tqdm

from .config import get_base_path, get_output_dir, get_aggregated_dir, DATASETS

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


def get_dataset_dirs(dataset: str, base_path: Optional[Path] = None) -> Tuple[Path, Path]:
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

def extract_metadata_from_filename(filename: str) -> Tuple[str, str, str]:
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


def find_all_performance_files(base_dir: str, targets: Optional[List[str]] = None) -> List[str]:
    """
    Find all performance files across all targets.

    Args:
        base_dir: Base directory containing target folders
        targets: Optional list of specific targets to process

    Returns:
        List of file paths to performance CSV files
    """
    if targets is None:
        targets = [d for d in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, d))]

    all_files = []
    for target in tqdm(targets, desc="Finding files"):
        target_dir = os.path.join(base_dir, target)
        performance_dir = os.path.join(target_dir, "results", "performance")

        if not os.path.exists(performance_dir):
            continue

        pattern = f"{target}_*_*_performance.csv"
        files = glob.glob(os.path.join(performance_dir, pattern))
        all_files.extend(files)

    return all_files


def process_file(file_path: str, metric_name: str, threshold: float) -> Optional[pl.DataFrame]:
    """
    Process a single performance file.

    Args:
        file_path: Path to the performance CSV file
        metric_name: Name of the metric column to extract
        threshold: Threshold value to filter by

    Returns:
        Polars DataFrame with processed data, or None if processing failed
    """
    try:
        if not os.path.exists(file_path):
            print(f"Warning: File does not exist: {file_path}")
            return None

        if os.path.getsize(file_path) == 0:
            print(f"Warning: Empty file: {file_path}")
            return None

        target_name, docking_method, selection_method = extract_metadata_from_filename(file_path)

        columns = ['scoring_function', 'threshold', 'combination',
                   'consensus_method', metric_name]

        try:
            df = pl.read_csv(
                file_path,
                ignore_errors=True
            ).filter(pl.col('threshold') == threshold)

            available_columns = set(df.columns)
            for col in columns:
                if col not in available_columns and col != metric_name:
                    df = df.with_columns(pl.lit(None).alias(col))

            if metric_name not in available_columns:
                print(f"Warning: Metric {metric_name} not found in {file_path}")
                return None

            df = df.select(columns)

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

        if df.height == 0:
            return None

        try:
            df = df.with_columns([
                pl.when(pl.col('combination').is_null())
                .then(pl.col('combination'))
                .otherwise(
                    pl.col('combination')
                    .str.split('+')
                    .list.eval(pl.element().sort())
                    .list.join('+')
                )
                .alias('combination')
            ])
        except Exception:
            pass

        try:
            df = df.with_columns([
                pl.when(pl.col('scoring_function').is_null() | (pl.col('scoring_function') == ""))
                .then(pl.col('combination'))
                .otherwise(pl.col('scoring_function'))
                .alias('scoring')
            ])
        except Exception:
            df = df.with_columns([
                pl.lit("unknown").alias('scoring')
            ])

        df = df.with_columns([
            pl.lit(target_name).alias('target'),
            pl.lit(docking_method).alias('docking_method'),
            pl.lit(selection_method).alias('selection_method')
        ])

        needed_columns = [
            'scoring', 'consensus_method',
            metric_name, 'target', 'docking_method', 'selection_method'
        ]

        for col in needed_columns:
            if col not in df.columns and col != metric_name:
                df = df.with_columns(pl.lit(None).alias(col))

        df = df.select(needed_columns)

        return df

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        print(traceback.format_exc())
        return None


def load_performance_data(
    base_dir: str,
    metric_name: str,
    threshold: float,
    targets: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Load performance data sequentially from all files.

    Args:
        base_dir: Base directory containing target folders
        metric_name: Name of the metric to load
        threshold: Threshold value to filter by
        targets: Optional list of specific targets to process

    Returns:
        Combined Polars DataFrame with all performance data
    """
    all_files = find_all_performance_files(base_dir, targets)

    if not all_files:
        raise ValueError("No performance files found.")

    print(f"Found {len(all_files)} performance files.")
    print(f"Processing files sequentially...")

    all_data = []

    for file_path in tqdm(all_files, desc="Processing files"):
        result = process_file(file_path, metric_name, threshold)
        if result is not None:
            all_data.append(result)

    if not all_data:
        raise ValueError("No data was loaded. Check file paths and formats.")

    try:
        combined_df = pl.concat(all_data)
    except Exception as e:
        print(f"Error during concatenation: {e}")
        print("Attempting schema-consistent concatenation...")

        all_columns = set()
        for df in all_data:
            all_columns.update(df.columns)

        normalized_dfs = []
        for df in all_data:
            missing_cols = all_columns - set(df.columns)
            if missing_cols:
                for col in missing_cols:
                    df = df.with_columns(pl.lit(None).alias(col))
            normalized_dfs.append(df)

        combined_df = pl.concat(normalized_dfs)

    print(f"Loaded {combined_df.height} rows with {metric_name} at threshold {threshold}.")

    return combined_df


def create_final_table_streaming(df: pl.DataFrame, metric_name: str) -> pl.DataFrame:
    """
    Create the final pivot table using a streaming approach.

    Args:
        df: Combined DataFrame with all performance data
        metric_name: Name of the metric column

    Returns:
        Pivot table with methods as rows and targets as columns
    """
    print("Creating final table...")

    df = df.with_columns([
        pl.col('consensus_method').fill_null(""),
    ])

    unique_rows = df.select([
        'docking_method', 'scoring', 'consensus_method', 'selection_method'
    ]).unique()

    print(f"Found {unique_rows.height} unique combinations of methods")

    unique_rows = unique_rows.rename({
        'docking_method': 'docking'
    })

    targets = list(set(df['target']))

    for target in tqdm(targets, desc="Processing targets"):
        target_df = df.filter(pl.col('target') == target)

        temp_df = target_df.select([
            'docking_method', 'scoring', 'consensus_method', 'selection_method',
            (pl.col(metric_name)).alias(target)
        ])

        temp_df = temp_df.rename({
            'docking_method': 'docking'
        })

        unique_rows = unique_rows.join(
            temp_df,
            on=['docking', 'scoring', 'consensus_method', 'selection_method'],
            how='left'
        )

    unique_rows = unique_rows.fill_null(0)

    return unique_rows


def save_pivot_table(
    df: pl.DataFrame,
    output_dir: str,
    file_prefix: str,
    metric_name: str,
    threshold: float
) -> Tuple[str, str]:
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

    csv_filename = f"{file_prefix}_{metric_name}_pivot_thresh{threshold_str}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    df.write_csv(csv_filepath)
    print(f"CSV saved to: {csv_filepath}")

    parquet_filename = f"{file_prefix}_{metric_name}_pivot_thresh{threshold_str}.parquet"
    parquet_filepath = os.path.join(output_dir, parquet_filename)
    df.write_parquet(parquet_filepath)
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


def run_aggregation_pipeline(
    base_dir: str,
    output_dir: str,
    file_prefix: str = FILE_PREFIX,
    metrics: Optional[List[str]] = None,
    thresholds: Optional[List[float]] = None,
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

    for metric in metrics:
        if metric in ["auc_roc", "bedroc"]:
            thresholds_to_use = [thresholds[0]]
        else:
            thresholds_to_use = thresholds

        for threshold in thresholds_to_use:
            if skip_existing and check_output_exists(output_dir, file_prefix, metric, threshold):
                print(f"\nSkipping metric={metric}, threshold={threshold} - output files already exist")
                continue

            print(f"\n{'='*80}")
            print(f"Starting analysis with metric={metric}, threshold={threshold}")
            print(f"{'='*80}")

            start_time = time.time()

            try:
                combined_df = load_performance_data(
                    base_dir,
                    metric,
                    threshold
                )
                data_load_time = time.time()
                print(f"Data loading time: {data_load_time - start_time:.2f} seconds")

                if verbose:
                    print("\nSample of combined data:")
                    print(combined_df.head(5))

                pivot_start_time = time.time()
                pivot_df = create_final_table_streaming(combined_df, metric)
                pivot_time = time.time()
                print(f"Pivot table creation time: {pivot_time - pivot_start_time:.2f} seconds")

                if verbose:
                    print("\nSample of pivot table:")
                    print(pivot_df.head(5))

                save_pivot_table(
                    pivot_df,
                    output_dir,
                    file_prefix,
                    metric,
                    threshold
                )
                save_time = time.time()
                print(f"Save time: {save_time - pivot_time:.2f} seconds")
                print(f"Total execution time for {metric} at threshold {threshold}: {save_time - start_time:.2f} seconds")

            except Exception as e:
                print(f"Error processing metric {metric} at threshold {threshold}: {e}")
                print(traceback.format_exc())

    total_end_time = time.time()
    print(f"\nTotal pipeline execution time: {total_end_time - total_start_time:.2f} seconds")
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


def parse_list_arg(arg_value: Optional[str], default: List) -> List:
    """Parse comma-separated argument into list."""
    if arg_value is None:
        return default
    return [x.strip() for x in arg_value.split(',')]


def parse_threshold_arg(arg_value: Optional[str]) -> Optional[List[float]]:
    """Parse comma-separated threshold argument into list of floats."""
    if arg_value is None:
        return None
    return [float(x.strip()) for x in arg_value.split(',')]


def main():
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
