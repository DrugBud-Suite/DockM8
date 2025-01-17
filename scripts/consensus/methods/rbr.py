"""This module provides a function to calculate the Rank by Rank (RbR) consensus score."""


import numpy as np
import pandas as pd


# Optimized RBR
def rbr_consensus(data: pd.DataFrame, columns: list[str], id_column: str = "ID") -> pd.DataFrame:
    """
    Calculate RBR score using numpy operations.
    Returns scores where higher values indicate better performance.
    """
    values = data[columns].values
    n_samples, n_cols = values.shape
    
    # Vectorized rank calculation - reversed to make higher values get higher ranks
    ranks = np.zeros_like(values, dtype=float)
    for i in range(n_cols):
        ranks[:, i] = (-values[:, i]).argsort().argsort() + 1
    
    mean_ranks = ranks.mean(axis=1)
    
    # Invert ranks so higher scores are better
    max_rank = n_samples
    inverted_ranks = max_rank - mean_ranks + 1
    
    return pd.DataFrame({
        id_column: data[id_column],
        'RbR': inverted_ranks
    })
