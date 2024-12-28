"""This module provides a function to calculate the Z-score consensus score."""


import pandas as pd


# Optimized Z-score
def zscore_consensus(data: pd.DataFrame, columns: list[str], id_column: str = "ID") -> pd.DataFrame:
    """Calculate Z-score using numpy operations."""
    values = data[columns].values
    z_scores = (values - values.mean(axis=0)) / values.std(axis=0)
    scores = z_scores.mean(axis=1)
    
    return pd.DataFrame({
        id_column: data[id_column],
        'Zscore': scores
    })
