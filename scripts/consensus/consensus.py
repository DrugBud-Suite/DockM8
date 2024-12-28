"""Optimized core functionality for consensus scoring implementation."""

from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from .utils.utils import load_data

if TYPE_CHECKING:
    from collections.abc import Callable
    
from .methods.ecr import ecr_consensus
from .methods.rbr import rbr_consensus
from .methods.zscore import zscore_consensus
from .methods.rbv_tiebreaker import rbv_consensus
from .methods.soft_rbv import soft_rbv_consensus
# Update method dictionary with new lowercase names
_METHODS = {
    "ecr": ecr_consensus,
    "rbr": rbr_consensus,
    "rbv": rbv_consensus,
    "zscore": zscore_consensus,
    #"pareto": pareto_consensus,
    "soft_rbv": soft_rbv_consensus
}

def load_and_validate_data(
    data: str | Path | pd.DataFrame,
    id_column: str,
    columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Optimized data loading and validation."""
    # Load data if it's a file path
    if isinstance(data, (str, Path)):
        data = load_data(data)
    
    if data.empty:
        raise ValueError("Input data is empty")
    
    if id_column not in data.columns:
        raise ValueError(f"ID column '{id_column}' not found in the data")
    
    # Pre-compute numeric columns once
    numeric_mask = data.dtypes.apply(pd.api.types.is_numeric_dtype)
    if columns:
        # Use numpy operations for column filtering
        valid_columns = [col for col in columns if col in data.columns and numeric_mask[col]]
        if not valid_columns:
            raise ValueError("None of the specified columns were found in the data or were numeric")
    else:
        valid_columns = data.columns[numeric_mask & (data.columns != id_column)].tolist()
    
    if not valid_columns:
        raise ValueError("No valid numeric columns found for scoring")
        
    # Optimize memory by selecting only needed columns
    return data[[id_column, *valid_columns]], valid_columns

def handle_nan_values(
    data: pd.DataFrame,
    valid_columns: list[str],
    nan_strategy: str
) -> pd.DataFrame:
    """Optimized NaN handling using vectorized operations."""
    # Extract numeric data for faster operations
    numeric_data = data[valid_columns].values
    
    if nan_strategy == "raise":
        if np.isnan(numeric_data).any():
            raise ValueError("Input data contains NaN values in scoring columns")
    elif nan_strategy == "drop":
        mask = ~np.isnan(numeric_data).any(axis=1)
        return data[mask]
    elif nan_strategy == "fill_mean":
        means = np.nanmean(numeric_data, axis=0)
        np.copyto(numeric_data, means, where=np.isnan(numeric_data))
    elif nan_strategy == "fill_median":
        medians = np.nanmedian(numeric_data, axis=0)
        np.copyto(numeric_data, medians, where=np.isnan(numeric_data))
    elif nan_strategy == "interpolate":
        # Use more efficient interpolation
        for i in range(numeric_data.shape[1]):
            col = numeric_data[:, i]
            mask = np.isnan(col)
            if mask.any():
                valid = ~mask
                indices = np.arange(len(col))
                col[mask] = np.interp(indices[mask], indices[valid], col[valid])
    else:
        raise ValueError(f"Invalid nan_strategy: {nan_strategy}")
    
    data = data.copy()
    data[valid_columns] = numeric_data
    return data

def apply_selected_methods(
    data: pd.DataFrame,
    valid_columns: list[str],
    id_column: str,
    selected_methods: list[Callable],
    normalize: bool = True,
    aggregation: str = "best",
) -> list[pd.DataFrame]:
    results = []
    
    for method in selected_methods:
        # Apply method and keep original result DataFrame
        result = method(data, valid_columns, id_column)
        
        # Handle duplicates if needed while preserving ID-score relationship
        if aggregation == "best":
            result = result.sort_values(result.columns[-1], ascending=False)
            result = result.drop_duplicates(subset=[id_column], keep='first')
        elif aggregation == "avg":
            result = result.groupby(id_column)[result.columns[-1]].mean().reset_index()
            
        # Normalize if requested while maintaining ID-score pairing
        if normalize and len(result) > 0:
            score_column = result.columns[-1]
            scores = result[score_column].values
            min_score, max_score = np.min(scores), np.max(scores)
            if max_score != min_score:
                result[score_column] = (scores - min_score) / (max_score - min_score)
            else:
                result[score_column] = np.zeros_like(scores)
        
        results.append(result)
    
    return results

def combine_results(results: list[pd.DataFrame], id_column: str) -> pd.DataFrame:
    """Optimized result combination using efficient merging."""
    if not results:
        return pd.DataFrame()
    
    # Use more efficient merge operation
    final_result = results[0]
    if len(results) > 1:
        for result in results[1:]:
            final_result = pd.merge(
                final_result,
                result,
                on=id_column,
                how='inner',
            )
    
    # # Optimize sorting
    # score_columns = [col for col in final_result.columns if col != id_column]
    # if score_columns:
    #     final_result = final_result.sort_values(
    #         by=score_columns,
    #         ascending=False,
    #         ignore_index=True
    #     )
    
    return final_result

def apply_consensus_scoring(
    data: str | Path | pd.DataFrame,
    methods: str | list[str] = "all",
    columns: list[str] | None = None,
    id_column: str = "ID",
    aggregation: str = "best",
    nan_strategy: str = "raise",
    output: str | Path | None = None,
    normalize: bool = True,
) -> pd.DataFrame | Path:
    """Optimized main consensus scoring function."""
    # Use optimized functions
    data, valid_columns = load_and_validate_data(data, id_column, columns)
    data = handle_nan_values(data, valid_columns, nan_strategy)
    selected_methods = select_methods(methods, _METHODS)
    
    results = apply_selected_methods(
        data=data,
        valid_columns=valid_columns,
        id_column=id_column,
        selected_methods=selected_methods,
        normalize=normalize,
        aggregation=aggregation,
    )
    
    final_result = combine_results(results, id_column)
    
    if output:
        return save_results(final_result, output)
    return final_result

def select_methods(methods: str | list[str], available_methods: dict) -> list[Callable]:
    """Select consensus methods to apply.

    Parameters
    ----------
    - methods: Union[str, List[str]]
    The consensus methods to apply. Can be 'all' or a list of method names.
    - available_methods: dict
    Dictionary of available methods.

    Returns:
    -------
    - List[Callable]
    List of selected method functions.
    """
    if methods == "all":
        selected_methods = list(available_methods.values())
    elif isinstance(methods, str):
        if methods not in available_methods:
            msg = f"Invalid method: {methods}"
            raise ValueError(msg)
        selected_methods = [available_methods[methods]]
    else:
        selected_methods = []
        for method in methods:
            if method not in available_methods:
                msg = f"Invalid method: {method}"
                raise ValueError(msg)
            selected_methods.append(available_methods[method])

    return selected_methods

def save_results(final_result: pd.DataFrame, output: str | Path) -> Path:
    """Save the final results to the specified output file.

    Parameters
    ----------
    - final_result: pd.DataFrame
    The DataFrame containing the final results.
    - output: Union[str, Path]
    File path to save the results.

    Returns:
    -------
    - Path
    The output file path.
    """
    output_path = Path(output)
    final_result.to_csv(output_path, index=False)
    return output_path
