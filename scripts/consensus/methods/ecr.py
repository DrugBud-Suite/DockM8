"""Exponential Consensus Ranking implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Optimized ECR
def ecr_consensus(data: pd.DataFrame, columns: list[str], id_column: str = "ID") -> pd.DataFrame:
    """Calculate the ECR consensus score using vectorized operations."""
    values = data[columns].values
    sigma = 0.05 * len(data)
    
    # Vectorized rank calculation
    temp = values.argsort(axis=0)
    ranks = np.empty_like(temp, dtype=float)
    for i in range(temp.shape[1]):
        ranks[temp[:,i], i] = np.arange(temp.shape[0], 0, -1)
    
    ecr_scores = np.exp(-ranks / sigma)
    scores = ecr_scores.sum(axis=1) / sigma
    
    return pd.DataFrame({
        id_column: data[id_column],
        'ECR': scores
    })
