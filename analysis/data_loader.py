"""
Data loading and validation for DockM8 analysis.

This module provides functions for loading performance data from CSV files,
including AVE split data (training/validation) and literature comparison data.
"""

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict

import pandas as pd
from tqdm import tqdm

from .config import NUMERIC_COLUMNS, THRESHOLD_METRIC_BASES
from .utils import create_workflow_id, clean_column_names


# =============================================================================
# AVE DATA VALIDATION
# =============================================================================

def validate_targets(
    targets: List[str],
    split_dir: Path,
    full_dir: Path
) -> List[str]:
    """
    Validate that targets have required directories and files.

    Args:
        targets: List of target names to validate
        split_dir: Directory containing split (train/validation) data
        full_dir: Directory containing full dataset results

    Returns:
        List of validated target names that have all required files
    """
    validated = []

    for target in targets:
        split_perf_dir = split_dir / target / "performance"
        full_perf_dir = full_dir / target / "results" / "performance"

        if not split_perf_dir.is_dir():
            print(f"  Warning: Split performance dir not found for {target}")
            continue
        if not full_perf_dir.is_dir():
            print(f"  Warning: Full performance dir not found for {target}")
            continue
        if not list(split_perf_dir.glob(f"{target}_*_performance_*.csv")):
            print(f"  Warning: No split CSVs found for {target}")
            continue
        if not list(full_perf_dir.glob(f"{target}_*_performance.csv")):
            print(f"  Warning: No full CSVs found for {target}")
            continue

        validated.append(target)

    return validated


def discover_targets(split_dir: Path) -> List[str]:
    """
    Discover available targets from split directory.

    Args:
        split_dir: Directory containing split data subdirectories

    Returns:
        Sorted list of target names found in the directory
    """
    targets = []
    if split_dir.is_dir():
        for item in split_dir.iterdir():
            if item.is_dir() and (item / "performance").is_dir():
                targets.append(item.name)
    return sorted(targets)


# =============================================================================
# AVE DATA LOADING
# =============================================================================

def load_target_data(
    target_name: str,
    split_dir: Path,
    full_dir: Path
) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Load AVE (train/valid) and full dataset performance data for a single target.

    Args:
        target_name: Name of the target to load
        split_dir: Directory containing split data
        full_dir: Directory containing full dataset results

    Returns:
        Tuple of (DataFrame with all data, list of warning messages)
    """
    data_list = []
    warnings = []

    split_perf_dir = split_dir / target_name / "performance"
    for f_path in split_perf_dir.glob(f"{target_name}_*_performance_*.csv"):
        try:
            if '_training.csv' in f_path.name:
                set_type = 'training'
            elif '_validation.csv' in f_path.name:
                set_type = 'validation'
            else:
                continue

            df = pd.read_csv(f_path, low_memory=False)
            df['target'] = target_name
            df['set'] = set_type
            df['filename_base'] = f_path.name.replace(f'_performance_{set_type}.csv', '')
            data_list.append(df)
        except Exception as e:
            warnings.append(f"Error reading {f_path.name}: {e}")

    full_perf_dir = full_dir / target_name / "results" / "performance"
    for f_path in full_perf_dir.glob(f"{target_name}_*_performance.csv"):
        if '_training.csv' in f_path.name or '_validation.csv' in f_path.name:
            continue
        try:
            df = pd.read_csv(f_path, low_memory=False)
            df['target'] = target_name
            df['set'] = 'full'
            df['filename_base'] = f_path.name.replace('_performance.csv', '')
            data_list.append(df)
        except Exception as e:
            warnings.append(f"Error reading {f_path.name}: {e}")

    if not data_list:
        return None, warnings + ["No data loaded for target."]

    df_all = pd.concat(data_list, ignore_index=True)

    for col in ['combination', 'consensus_method', 'scoring_function', 'filename_base']:
        if col not in df_all.columns:
            df_all[col] = pd.NA

    df_all['workflow_id'] = df_all.apply(create_workflow_id, axis=1)

    return df_all, warnings


def load_all_data(
    targets: List[str],
    split_dir: Path,
    full_dir: Path,
    max_workers: Optional[int] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Load data for all targets in parallel.

    Args:
        targets: List of target names to load
        split_dir: Directory containing split data
        full_dir: Directory containing full dataset results
        max_workers: Maximum parallel workers (default: CPU count)

    Returns:
        Tuple of (combined DataFrame, dict of warnings by target)
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    all_dfs = []
    all_warnings = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(load_target_data, t, split_dir, full_dir): t
            for t in targets
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading data"):
            target_name = futures[future]
            try:
                df_result, warns = future.result()
                if df_result is not None:
                    all_dfs.append(df_result)
                if warns:
                    all_warnings[target_name] = warns
            except Exception as e:
                all_warnings[target_name] = [f"Future error: {e}"]

    if all_dfs:
        df_combined = pd.concat(all_dfs, ignore_index=True)
    else:
        df_combined = pd.DataFrame()

    return df_combined, all_warnings


def validate_metrics(df: pd.DataFrame, requested_metrics: List[str]) -> List[str]:
    """
    Validate that requested metrics exist in the dataframe.

    Args:
        df: DataFrame to check for metrics
        requested_metrics: List of metric names to validate

    Returns:
        List of valid metric names that exist in the DataFrame
    """
    available_cols = df.columns.tolist()
    validated = []

    for metric in requested_metrics:
        if metric in available_cols:
            validated.append(metric)
        elif metric in THRESHOLD_METRIC_BASES:
            validated.append(metric)
        else:
            print(f"  Warning: Metric '{metric}' not found. Skipping.")

    return validated


# =============================================================================
# LITERATURE DATA LOADING
# =============================================================================

def load_literature_source(
    file_path: Path,
    source_name: str,
    numeric_cols: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Load a single literature or DockM8 source CSV file.

    Args:
        file_path: Path to the CSV file
        source_name: Name of the source ('Literature' or 'DockM8')
        numeric_cols: List of columns to convert to numeric

    Returns:
        Cleaned DataFrame with Source column added, or None if loading fails
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLUMNS

    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df = clean_column_names(df)
        df['Source'] = source_name
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading {source_name} data from {file_path}: {e}")
        return None


def load_literature_dataset(
    dataset_name: str,
    literature_path: Path,
    dockm8_path: Path,
    numeric_cols: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Load and combine literature and DockM8 data for a single dataset.

    Args:
        dataset_name: Name of the dataset (for logging)
        literature_path: Path to literature CSV file
        dockm8_path: Path to DockM8 CSV file
        numeric_cols: List of columns to convert to numeric

    Returns:
        Combined DataFrame or None if no data loaded
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLUMNS

    df_list = []

    lit_df = load_literature_source(literature_path, 'Literature', numeric_cols)
    if lit_df is not None:
        df_list.append(lit_df)

    dockm8_df = load_literature_source(dockm8_path, 'DockM8', numeric_cols)
    if dockm8_df is not None:
        df_list.append(dockm8_df)

    if df_list:
        combined = pd.concat(df_list, ignore_index=True)
        print(f"Loaded {dataset_name}: {len(combined)} total rows")
        return combined
    else:
        print(f"Warning: No data loaded for dataset {dataset_name}")
        return None


def load_all_literature_datasets(
    dataset_info: Dict[str, Dict[str, Path]],
    numeric_cols: Optional[List[str]] = None
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load all literature datasets from the provided configuration.

    Args:
        dataset_info: Dictionary mapping dataset names to source paths.
            Format: {
                'DEKOIS': {
                    'Literature': Path('...'),
                    'DockM8': Path('...')
                },
                ...
            }
        numeric_cols: List of columns to convert to numeric

    Returns:
        Dictionary mapping dataset names to DataFrames (or None if loading failed)
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLUMNS

    all_data = {}
    for dataset_name, sources in dataset_info.items():
        literature_path = sources.get('Literature')
        dockm8_path = sources.get('DockM8')

        if literature_path is None or dockm8_path is None:
            print(f"Warning: Missing path for {dataset_name}")
            all_data[dataset_name] = None
            continue

        all_data[dataset_name] = load_literature_dataset(
            dataset_name=dataset_name,
            literature_path=Path(literature_path),
            dockm8_path=Path(dockm8_path),
            numeric_cols=numeric_cols
        )

    return all_data
