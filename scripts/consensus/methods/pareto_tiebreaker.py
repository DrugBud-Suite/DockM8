"""Pareto ranking consensus scoring implementation."""

from __future__ import annotations

import pandas as pd
from paretoset import paretoset

def calculate_zscore(data: pd.DataFrame, columns: list[str], id_column: str = "ID") -> pd.DataFrame:
    """Calculate pure Z-scores."""
    values = data[columns].values
    z_scores = (values - values.mean(axis=0)) / values.std(axis=0)
    zscore_avg = z_scores.mean(axis=1)
    
    return pd.DataFrame({
        id_column: data[id_column],
        'Zscore': zscore_avg
    })

def calculate_pareto(data: pd.DataFrame, columns: list[str], id_column: str = "ID") -> pd.DataFrame:
    """Calculate pure Pareto ranks."""
    scoring_data = data[[id_column, *columns]].copy()
    sense = ["max"] * len(columns)
    
    scoring_data["Pareto"] = 0
    rank = 1
    remaining_data = scoring_data.copy()
    
    while not remaining_data.empty:
        mask = paretoset(remaining_data[columns], sense=sense, use_numba=True)
        current_ids = remaining_data.loc[mask, id_column]
        scoring_data.loc[scoring_data[id_column].isin(current_ids), "Pareto"] = rank
        remaining_data = remaining_data.loc[~mask]
        rank += 1
    
    # Invert ranks so highest rank is best
    max_rank = scoring_data["Pareto"].max()
    scoring_data["Pareto"] = max_rank + 1 - scoring_data["Pareto"]
    
    return scoring_data[[id_column, "Pareto"]]

def pareto_consensus(data: pd.DataFrame, columns: list[str], id_column: str = "ID") -> pd.DataFrame:
    """Combine Pareto and Z-score results with proper normalization."""
    # Get individual scores
    pareto_scores = calculate_pareto(data, columns, id_column)
    z_scores = calculate_zscore(data, columns, id_column)
    
    # Merge on ID
    combined = pd.merge(pareto_scores, z_scores, on=id_column, how='inner')
    
    # Normalize z-scores to 0-0.999 range for tiebreaking
    ranks = combined['Zscore'].rank(method='dense') - 1  # -1 to start at 0
    normalized_tiebreaker = ranks / len(ranks)
    
    # Combine scores
    combined['Pareto'] = combined['Pareto'] + normalized_tiebreaker
    
    return combined[[id_column, 'Pareto']].sort_values('Pareto', ascending=False)
