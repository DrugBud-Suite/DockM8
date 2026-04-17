"""
Centralized configuration and constants for DockM8 analysis scripts.

This module consolidates all configuration constants, path management, and shared
helper functions that were previously scattered across analysis_config.py and utils.py.
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import polars as pl


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

def get_base_path(cli_path: Optional[Path] = None) -> Path:
    """
    Get base path to raw benchmark data (Zenodo download).

    Args:
        cli_path: Path provided via --base-path CLI argument

    Returns:
        Path object to base results directory
    """
    if cli_path is not None:
        return cli_path
    return Path(__file__).parent.parent


def get_output_dir(cli_path: Optional[Path] = None) -> Path:
    """
    Get output directory for generated results.

    Args:
        cli_path: Path provided via --output-dir CLI argument

    Returns:
        Path object to output directory (default: results/output/)
    """
    if cli_path is not None:
        return cli_path
    return Path(__file__).parent.parent / "results" / "output"


def get_literature_dir() -> Path:
    """Return path to committed literature baseline CSVs."""
    return Path(__file__).parent / "data" / "literature"


def get_aggregated_dir(output_dir: Optional[Path] = None) -> Path:
    """Return path to aggregated parquet pivot tables."""
    if output_dir is None:
        output_dir = get_output_dir()
    return output_dir / "aggregated"


def get_dockm8_results_dir(output_dir: Optional[Path] = None) -> Path:
    """Return path to extracted dockm8 comparison CSVs."""
    if output_dir is None:
        output_dir = get_output_dir()
    return output_dir / "dockm8_results"


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

DATASETS = ["DEKOIS", "DUD-E", "Lit-PCBA"]
DATASET_DIRS = {"DEKOIS": "DEKOIS", "DUD-E": "DUD-E", "Lit-PCBA": "Lit-PCBA"}

# =============================================================================
# METRICS CONFIGURATION
# =============================================================================

METRICS = ["ref", "ef", "auc_roc", "bedroc", "pm"]
THRESHOLDS = ["0p1", "0p5", "1", "5"]
THRESH_MAP = {"0p1": "0.1%", "0p5": "0.5%", "1": "1%", "5": "5%"}

THRESHOLD_METRIC_BASES = {'ef', 'ref', 'roce', 'pm', 'mcc', 'ccr', 'ckc'}

METRIC_DISPLAY_NAMES = {
    'auc_roc': 'AUC ROC',
    'aupr': 'AUPR',
    'bedroc': 'BEDROC',
    'ef': 'EF',
    'ref': 'REF',
    'pm': 'Power Metric',
    'roce': 'ROC Enrichment',
    'ccr': 'CCR',
    'mcc': 'MCC',
    'ckc': 'CKC'
}

NUMERIC_COLUMNS = [
    'EF_lit', 'NEF_lit', 'EF_calc', 'NEF_calc',
    'n', 'N', 'Ns', 'ns', 'PM', 'ROCE', 'CCR',
    'REF', 'MCC', 'CKC', 'AUC_ROC', 'BEDROC'
]

# =============================================================================
# WORKFLOW CONFIGURATION
# =============================================================================

WORKFLOW_COLS = ['docking', 'scoring', 'consensus_method', 'selection_method']
SINGLE_SF_PLACEHOLDER = 'Single SF'

# =============================================================================
# SCORING FUNCTION CATEGORIES
# =============================================================================

SF_CATEGORIES = {
    'ml': {
        'color': 'royalblue',
        'functions': [
            'RTMScore', 'NNScore', 'CNN-Affinity', 'RFScoreVS',
            'GenScore-scoring', 'GenScore-balanced', 'GenScore-docking', 'CNN-Score'
        ]
    },
    'empirical': {
        'color': 'mediumseagreen',
        'functions': ['AD4', 'LinF9', 'CHEMPLP', 'Vinardo', 'GNINA-Affinity']
    },
    'knowledge': {
        'color': 'lightcoral',
        'functions': ['KORP-PL', 'ConvexPLR']
    },
    'consensus': {
        'color': '#ff7700',
        'functions': []
    },
    'other': {
        'color': 'darkgray',
        'functions': []
    }
}


# =============================================================================
# SCORING FUNCTION HELPERS
# =============================================================================

def get_sf_color(sf_name: str) -> str:
    """
    Determine color based on scoring function name.

    Args:
        sf_name: Name of the scoring function

    Returns:
        Color string for the scoring function category
    """
    for category, details in SF_CATEGORIES.items():
        if any(func == sf_name for func in details['functions']):
            return details['color']
    for category, details in SF_CATEGORIES.items():
        if any(func in sf_name for func in details['functions']):
            return details['color']
    return SF_CATEGORIES['other']['color']


def get_sf_type_and_color(
    model_name: str,
    categories: Optional[Dict] = None
) -> Tuple[str, str]:
    """
    Determines the scoring function type and color for a given model name.

    Checks for consensus prefix first ('dockm8-'), then extracts the SF name
    (part before '/') and checks against category lists.

    Args:
        model_name: The model name string
        categories: Dictionary of SF categories (default: SF_CATEGORIES)

    Returns:
        Tuple of (sf_type, color)
    """
    if categories is None:
        categories = SF_CATEGORIES

    model_name_lower = model_name.lower()

    if 'dockm8-' in model_name_lower:
        return 'consensus', categories['consensus']['color']

    sf_name_part = model_name.split('/')[0] if '/' in model_name else model_name

    for sf_type, details in categories.items():
        if sf_type in ['consensus', 'other']:
            continue
        if sf_name_part in details['functions']:
            return sf_type, details['color']

    return 'other', categories['other']['color']


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_metric_data(
    aggregated_dir: Path,
    dataset: str,
    metric: str,
    threshold: str
) -> Optional[pd.DataFrame]:
    """
    Load parquet file for a given dataset/metric/threshold combination.

    Args:
        aggregated_dir: Path to aggregated output directory (results/output/aggregated/)
        dataset: Dataset name (e.g., 'DEKOIS', 'DUD-E', 'Lit-PCBA')
        metric: Metric name (e.g., 'ref', 'ef', 'auc_roc')
        threshold: Threshold string (e.g., '0p1', '1', '5')

    Returns:
        DataFrame or None if file not found
    """
    file_stem = f"combined_results_{metric}_pivot_thresh{threshold}"
    parquet_file_path = aggregated_dir / f"{dataset}_aggregated_results" / f"{file_stem}.parquet"

    try:
        df_pl = pl.read_parquet(parquet_file_path)
        df = pd.DataFrame(df_pl.to_dict())
        return df
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading file {parquet_file_path}: {e}")
        return None


def load_all_datasets(
    aggregated_dir: Path,
    metric: str,
    threshold: str,
    datasets: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load data for all datasets for a given metric/threshold.

    Args:
        aggregated_dir: Path to aggregated output directory (results/output/aggregated/)
        metric: Metric name
        threshold: Threshold string
        datasets: List of datasets to load (default: all)
        verbose: Print loading status messages

    Returns:
        Dictionary mapping dataset names to DataFrames (or None if not found)
    """
    if datasets is None:
        datasets = DATASETS

    loaded_data = {}
    for ds_name in datasets:
        df = load_metric_data(aggregated_dir, ds_name, metric, threshold)
        if verbose:
            if df is not None:
                print(f"  Loaded {ds_name} ({len(df)} rows)")
            else:
                print(f"  Warning: Could not load data for {ds_name}")
        loaded_data[ds_name] = df

    return loaded_data


# =============================================================================
# DATAFRAME HELPER FUNCTIONS
# =============================================================================

def calculate_overall_performance(
    df: pd.DataFrame,
    use_median: bool = False
) -> pd.DataFrame:
    """
    Calculates an overall performance metric for each workflow row.

    Args:
        df: DataFrame with workflow columns and target performance columns
        use_median: If True, use median instead of mean

    Returns:
        DataFrame with new 'overall_performance' column added
    """
    target_cols = [col for col in df.columns if col not in WORKFLOW_COLS]
    if not target_cols:
        df['overall_performance'] = np.nan
        return df

    if use_median:
        df['overall_performance'] = df[target_cols].apply(np.nanmedian, axis=1)
    else:
        df['overall_performance'] = df[target_cols].mean(axis=1)

    return df


def get_top_workflows_pd(
    df: pd.DataFrame,
    percentile: Optional[float],
    ranking_col: str = 'overall_performance'
) -> pd.DataFrame:
    """
    Filters the pandas DataFrame to keep only the top N percentile workflows.

    Args:
        df: Input DataFrame with workflow data
        percentile: Percentile threshold (0-100). None returns full DataFrame.
        ranking_col: Column name to use for ranking

    Returns:
        Filtered DataFrame containing only top percentile workflows
    """
    if df is None or percentile is None or df.empty:
        return df

    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    if percentile == 0:
        return df.iloc[0:0]

    if ranking_col not in df.columns:
        print(f"Error: Ranking column '{ranking_col}' not found. Returning empty DataFrame.")
        return df.iloc[0:0]

    if not pd.api.types.is_numeric_dtype(df[ranking_col]):
        print(f"Error: Ranking column '{ranking_col}' is not numeric. Cannot calculate percentile.")
        return df.iloc[0:0]

    valid_rankings = df[ranking_col].dropna()
    if valid_rankings.empty:
        print(f"Warning: No valid (non-NaN) values found in ranking column '{ranking_col}'. Returning empty DataFrame.")
        return df.iloc[0:0]

    threshold_value = valid_rankings.quantile((100.0 - percentile) / 100.0, interpolation='higher')

    if pd.isna(threshold_value):
        if percentile == 100:
            threshold_value = valid_rankings.min()
        else:
            print(f"Warning: Quantile calculation resulted in NaN for percentile {percentile} in column '{ranking_col}'. Returning empty DataFrame.")
            return df.iloc[0:0]

    df_top = df[(df[ranking_col] >= threshold_value) & (~df[ranking_col].isna())].copy()
    return df_top


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_list_arg(arg_value: Optional[str], default: List[str]) -> List[str]:
    """
    Parse comma-separated argument string into list.

    Args:
        arg_value: Comma-separated string or None
        default: Default list if arg_value is None

    Returns:
        List of parsed values
    """
    if arg_value is None:
        return default
    return [x.strip() for x in arg_value.split(',')]


def setup_argument_parser(description: str):
    """
    Create base argument parser with common arguments.

    Args:
        description: Script description

    Returns:
        Configured ArgumentParser
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--base-path',
        type=Path,
        default=None,
        help='Base path to raw benchmark data (Zenodo download)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: results/output/)'
    )

    parser.add_argument(
        '--metrics',
        type=str,
        default=None,
        help=f'Comma-separated list of metrics (default: {",".join(METRICS)})'
    )

    parser.add_argument(
        '--thresholds',
        type=str,
        default=None,
        help=f'Comma-separated list of thresholds (default: {",".join(THRESHOLDS)})'
    )

    parser.add_argument(
        '--datasets',
        type=str,
        default=None,
        help=f'Comma-separated list of datasets (default: {",".join(DATASETS)})'
    )

    return parser
