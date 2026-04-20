"""
Utility functions for DockM8 analysis scripts.

This module contains pure utility functions for data processing, formatting,
and correlation calculations. Constants have been moved to config.py.
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from typing import Tuple, Optional, List

from .config import THRESHOLD_METRIC_BASES, METRIC_DISPLAY_NAMES


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove whitespace from DataFrame column names.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with cleaned column names
    """
    df.columns = df.columns.str.strip()
    return df


def sanitize_filename(name: str) -> str:
    """
    Remove problematic characters from filenames.

    Replaces '%' with 'pct', '@' with 'at', spaces with '_',
    and removes any remaining non-alphanumeric characters.

    Args:
        name: Original filename string

    Returns:
        Sanitized filename safe for filesystem use
    """
    name = name.replace('%', 'pct').replace('@', 'at').replace(' ', '_')
    name = re.sub(r'[^\w\-_.]+', '', name)
    return name


def format_metric_name(metric_name: str, threshold: Optional[float] = None) -> str:
    """
    Format metric names for display with optional threshold.

    Args:
        metric_name: Raw metric name (e.g., 'auc_roc', 'ref')
        threshold: Optional threshold value for threshold-dependent metrics

    Returns:
        Formatted display name (e.g., 'AUC ROC', 'REF @ 1.0% (%)')
    """
    base_name = metric_name.split('_')[0].lower()
    display_name = METRIC_DISPLAY_NAMES.get(base_name, metric_name.upper())

    if base_name in ['ef', 'ref']:
        thresh_str = f" @ {threshold}%" if threshold is not None else ""
        display_name = f"{display_name}{thresh_str} (%)"
    elif threshold is not None and base_name in THRESHOLD_METRIC_BASES:
        display_name = f"{display_name} @ {threshold}%"

    return display_name


def create_workflow_id(row: pd.Series) -> str:
    """
    Create unique workflow identifier from row data.

    Constructs a workflow ID from the filename_base, docking method,
    selection method, and scoring function or consensus method.

    Args:
        row: DataFrame row with workflow information

    Returns:
        Unique workflow identifier string
    """
    if pd.isna(row.get('filename_base')):
        return "Unknown_FilenameBase"

    parts = row['filename_base'].split('_')
    docking_method = parts[1] if len(parts) > 1 else "UnknownDocking"
    selection_method = parts[2] if len(parts) > 2 else "UnknownSelection"

    if pd.notna(row.get('scoring_function')):
        return f"{docking_method}_{selection_method}_{row['scoring_function']}"
    elif pd.notna(row.get('combination')) and pd.notna(row.get('consensus_method')):
        return f"{docking_method}_{selection_method}_{row['combination']}_({row['consensus_method']})"
    else:
        return f"{docking_method}_{selection_method}_UnknownScoringType"


def calculate_correlation_metrics(
    x_values: np.ndarray,
    y_values: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate Spearman rho, Kendall tau, and RMSE between two arrays.

    Args:
        x_values: First array of values
        y_values: Second array of values

    Returns:
        Tuple of (spearman_rho, kendall_tau, rmse)
        Returns np.nan for any metric that cannot be calculated
    """
    spearman_rho, kendall_tau_val, rmse_val = np.nan, np.nan, np.nan

    valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
    x_corr = x_values[valid_indices]
    y_corr = y_values[valid_indices]

    if len(x_corr) >= 2:
        try:
            spearman_rho, _ = spearmanr(x_corr, y_corr)
            rmse_val = np.sqrt(np.mean((x_corr - y_corr) ** 2))
            kendall_tau_val, _ = kendalltau(x_corr, y_corr)
        except Exception:
            pass

    return spearman_rho, kendall_tau_val, rmse_val


def format_correlation_string(
    rho: float,
    tau: float,
    rmse: float,
    prefix: str = ""
) -> str:
    """
    Format correlation metrics into a display string.

    Args:
        rho: Spearman rho value
        tau: Kendall tau value
        rmse: RMSE value
        prefix: Optional prefix string

    Returns:
        Formatted string like "rho=0.85, tau=0.72, RMSE=0.15" or "N/A" if all NaN
    """
    parts = []
    if not np.isnan(rho):
        parts.append(f"rho={rho:.2f}")
    if not np.isnan(tau):
        parts.append(f"tau={tau:.2f}")
    if not np.isnan(rmse):
        parts.append(f"RMSE={rmse:.2f}")

    if not parts:
        return f"{prefix}N/A"
    return f"{prefix}{', '.join(parts)}"


def is_threshold_metric(metric: str) -> bool:
    """
    Check if a metric requires threshold filtering.

    Args:
        metric: Metric name

    Returns:
        True if the metric is threshold-dependent (ef, ref, roce, pm, mcc, ccr, ckc)
    """
    return metric.lower() in THRESHOLD_METRIC_BASES


def parse_ranking_metrics(ranking_str: str) -> List[Tuple[str, Optional[float]]]:
    """
    Parse ranking metrics string into list of (metric, threshold) tuples.

    Format: 'ref:1.0,bedroc' -> [('ref', 1.0), ('bedroc', None)]

    Args:
        ranking_str: Comma-separated string of metrics with optional thresholds

    Returns:
        List of (metric_name, threshold) tuples
    """
    results = []
    for item in ranking_str.split(','):
        item = item.strip()
        if ':' in item:
            metric, threshold = item.split(':', 1)
            results.append((metric.strip(), float(threshold)))
        else:
            results.append((item, None))
    return results
