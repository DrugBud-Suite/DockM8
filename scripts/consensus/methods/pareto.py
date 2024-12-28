"""Pareto ranking consensus scoring implementation."""

from __future__ import annotations

import pandas as pd
from paretoset import paretoset


def pareto_consensus(data: pd.DataFrame, columns: list[str], id_column: str = "ID") -> pd.DataFrame:
    """
    Calculate Pareto ranking consensus score.
    Returns scores from highest (best) to 1 (worst).
    """
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
    
    # Invert ranks so highest rank is best and lowest is 1
    max_rank = scoring_data["Pareto"].max()
    scoring_data["Pareto"] = max_rank + 1 - scoring_data["Pareto"]
    
    # If you prefer 0-1 scaling instead, uncomment this line:
    # scoring_data["Pareto"] = (scoring_data["Pareto"] - 1) / (max_rank - 1)
    
    return scoring_data[[id_column, "Pareto"]]
