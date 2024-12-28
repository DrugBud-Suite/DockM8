"""This module contains methods for calculating the Rank by Vote (RbV) consensus score."""


import numpy as np
import pandas as pd


# Optimized RBV
def rbv_consensus(data: pd.DataFrame, columns: list[str], id_column: str = "ID") -> pd.DataFrame:
    """Calculate RBV score using numpy operations."""
    values = data[columns].values
    threshold = np.percentile(values, 95, axis=0)
    scores = (values > threshold).sum(axis=1)
    
    return pd.DataFrame({
        id_column: data[id_column],
        'RbV': scores
    })
