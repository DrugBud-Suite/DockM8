"""
Workflow ranking and top performer identification.

This module provides functions for ranking workflows by performance metrics
and identifying top performers at various percentile levels.
"""

import math
from typing import Set, Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np

from .config import THRESHOLD_METRIC_BASES


def is_threshold_metric(metric: str) -> bool:
    """
    Check if a metric requires threshold filtering.

    Args:
        metric: Metric name

    Returns:
        True if the metric is threshold-dependent
    """
    return metric.lower() in THRESHOLD_METRIC_BASES


def rank_workflows(
    df: pd.DataFrame,
    metric: str,
    threshold: Optional[float] = None,
    set_type: str = 'training'
) -> pd.Series:
    """
    Calculate median scores per workflow for ranking.

    Args:
        df: Performance dataframe with 'set', 'workflow_id', and metric columns
        metric: The metric column to use for ranking
        threshold: If metric is threshold-dependent, filter to this threshold
        set_type: Which set to use for ranking ('training', 'validation', 'full')

    Returns:
        Series with workflow_id as index and median scores as values, sorted descending
    """
    if 'set' not in df.columns:
        print(f"  Warning: 'set' column missing in dataframe")
        return pd.Series(dtype=float)

    if 'workflow_id' not in df.columns:
        print(f"  Warning: 'workflow_id' column missing in dataframe")
        return pd.Series(dtype=float)

    df_filtered = df[df['set'] == set_type].copy()

    if is_threshold_metric(metric) and threshold is not None:
        if 'threshold' not in df_filtered.columns:
            print(f"  Warning: 'threshold' column missing for {metric}")
            return pd.Series(dtype=float)

        df_filtered['threshold'] = pd.to_numeric(df_filtered['threshold'], errors='coerce')
        df_filtered = df_filtered[df_filtered['threshold'] == threshold]

    if metric not in df_filtered.columns:
        print(f"  Error: Metric '{metric}' not found in data")
        return pd.Series(dtype=float)

    if df_filtered.empty:
        return pd.Series(dtype=float)

    df_filtered[metric] = pd.to_numeric(df_filtered[metric], errors='coerce')
    median_scores = df_filtered.groupby('workflow_id')[metric].median()
    median_scores = median_scores.dropna().sort_values(ascending=False)

    return median_scores


def get_top_percentile_workflows(
    scores: pd.Series,
    percentile: float
) -> Set[str]:
    """
    Get workflow IDs in the top percentile.

    Args:
        scores: Series with workflow_id as index, sorted descending
        percentile: Percentage of top workflows to select (e.g., 10.0 for top 10%)

    Returns:
        Set of workflow IDs in the top percentile
    """
    if scores.empty:
        return set()

    n_workflows = len(scores)
    n_top = max(1, math.ceil(n_workflows * percentile / 100.0))

    return set(scores.head(n_top).index)


def identify_top_workflows(
    df: pd.DataFrame,
    metric: str,
    threshold: Optional[float] = None,
    percentiles: Tuple[float, ...] = (10.0, 1.0, 0.1),
    set_type: str = 'training'
) -> Dict[str, Any]:
    """
    Identify top performing workflows at multiple percentile levels.

    Args:
        df: Performance dataframe
        metric: Metric to rank by
        threshold: Threshold for threshold-dependent metrics
        percentiles: Tuple of percentile levels to compute
        set_type: Which set to use for ranking

    Returns:
        Dictionary with:
            - 'scores': Series of median scores
            - 'top_N': Set of workflow IDs for each percentile N
    """
    scores = rank_workflows(df, metric, threshold, set_type)

    result = {'scores': scores}

    for pct in percentiles:
        key = f'top_{pct}'.replace('.', '_')
        result[key] = get_top_percentile_workflows(scores, pct)

    return result


def create_workflow_ranking_df(
    scores: pd.Series,
    metric: str,
    threshold: Optional[float],
    percentiles: Dict[str, Set[str]]
) -> pd.DataFrame:
    """
    Create a DataFrame with workflow rankings and percentile membership.

    Args:
        scores: Series of median scores indexed by workflow_id
        metric: The ranking metric used
        threshold: The threshold used (if any)
        percentiles: Dict mapping percentile names to sets of workflow IDs

    Returns:
        DataFrame with workflow_id, score, rank, and percentile columns
    """
    if scores.empty:
        return pd.DataFrame()

    df = pd.DataFrame({
        'workflow_id': scores.index,
        'score': scores.values,
        'rank': range(1, len(scores) + 1),
        'ranking_metric': metric,
        'threshold': threshold
    })

    for pct_name, wf_ids in percentiles.items():
        df[pct_name] = df['workflow_id'].isin(wf_ids)

    df['is_single_scoring'] = ~df['workflow_id'].str.contains(r'\(', regex=True)

    return df
