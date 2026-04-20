"""Optimized core functionality for consensus scoring implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from .methods.ecr import ecr_consensus
from .methods.rbr import rbr_consensus
from .methods.rbv_tiebreaker import rbv_consensus
from .methods.soft_rbv import soft_rbv_consensus
from .methods.zscore import zscore_consensus
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS

# Method dictionary with lowercase names
_METHODS = {
    "ecr": ecr_consensus,
    "rbr": rbr_consensus,
    "rbv": rbv_consensus,
    "zscore": zscore_consensus,
    "softrbv": soft_rbv_consensus
}

def load_and_validate_data(
    data: str | Path | pd.DataFrame,
    id_column: str,
    columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Optimized data loading and validation."""
    # Load data if it's a file path
    if isinstance(data, (str, Path)):
        data = pd.read_csv(data)
    
    if data.empty:
        raise ValueError("Input data is empty")
    
    if id_column not in data.columns:
        raise ValueError(f"ID column '{id_column}' not found in the data")
    
    # Vectorized check for numeric columns
    numeric_mask = np.array([pd.api.types.is_numeric_dtype(data[col].dtype) for col in data.columns])
    column_array = np.array(data.columns)
    
    if columns:
        # Use numpy operations for column filtering
        columns_set = set(columns)
        mask = np.array([col in columns_set and pd.api.types.is_numeric_dtype(data[col].dtype)
                        for col in data.columns])
        valid_columns = column_array[mask].tolist()
        if not valid_columns:
            raise ValueError("None of the specified columns were found in the data or were numeric")
    else:
        # Get all numeric columns except ID column
        mask = numeric_mask & (column_array != id_column)
        valid_columns = column_array[mask].tolist()
    
    if not valid_columns:
        raise ValueError("No valid numeric columns found for scoring")
        
    # Optimize memory by selecting only needed columns
    return data[[id_column] + valid_columns], valid_columns

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
        # Vectorized mean calculation and filling
        means = np.nanmean(numeric_data, axis=0)
        for i, col in enumerate(valid_columns):
            mask = np.isnan(numeric_data[:, i])
            if mask.any():
                data.loc[data.index[mask], col] = means[i]
    elif nan_strategy == "fill_median":
        # Vectorized median calculation and filling
        medians = np.nanmedian(numeric_data, axis=0)
        for i, col in enumerate(valid_columns):
            mask = np.isnan(numeric_data[:, i])
            if mask.any():
                data.loc[data.index[mask], col] = medians[i]
    elif nan_strategy == "interpolate":
        # Use pandas built-in interpolate which is optimized
        data[valid_columns] = data[valid_columns].interpolate(method='linear', axis=0)
    else:
        raise ValueError(f"Invalid nan_strategy: {nan_strategy}")
    
    return data

def normalize_scores(scores: np.ndarray, scoring_info: dict) -> np.ndarray:
    """
    Normalize scores to [0,1] range where 1 is always best.
    Optimized implementation that handles edge cases efficiently.
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.full_like(scores, 0.5)
        
    normalized = (scores - min_score) / (max_score - min_score)
    if scoring_info["best_value"] == "min":
        normalized = 1 - normalized
    return normalized

def normalize_input_data(
    data: pd.DataFrame,
    valid_columns: list[str],
    id_column: str
) -> pd.DataFrame:
    """Normalize input data based on rescoring function properties.
    Only keeps columns that are in RESCORING_FUNCTIONS and the ID column.
    """
    # First filter to only keep columns that are in RESCORING_FUNCTIONS
    rescoring_columns = [col for col in valid_columns if col in RESCORING_FUNCTIONS]
    
    if not rescoring_columns:
        raise ValueError("No valid rescoring function columns found for consensus")
    
    # Create a new DataFrame with only the ID column and rescoring function columns
    normalized_data = pd.DataFrame({id_column: data[id_column]})
    
    # Normalize each column based on its rescoring function
    for col in rescoring_columns:
        scoring_info = RESCORING_FUNCTIONS[col]
        # Convert to numeric values, handling string values
        scores = pd.to_numeric(data[col], errors='coerce').values
        normalized_data[col] = normalize_scores(scores, scoring_info)
    
    return normalized_data, rescoring_columns

def apply_consensus_scoring(
    data: str | Path | pd.DataFrame,
    method: str,
    columns: list[str] | None = None,
    id_column: str = "ID",
    aggregation: str = "best",
    nan_strategy: str = "raise",
    output: str | Path | None = None,
    normalize: bool = True,
) -> pd.DataFrame | Path:
    """Simplified consensus scoring function for a single method."""
    # Load and validate data
    data_df, valid_columns = load_and_validate_data(data, id_column, columns)
    
    # Handle NaN values
    data_df = handle_nan_values(data_df, valid_columns, nan_strategy)
    
    # Normalize input data before consensus - this happens every time
    # Only keep columns that are in RESCORING_FUNCTIONS
    normalized_df, rescoring_columns = normalize_input_data(data_df, valid_columns, id_column)
    
    # Select method
    if method not in _METHODS:
        raise ValueError(f"Invalid method: {method}")
    
    consensus_method = _METHODS[method]
    
    # Apply the consensus method
    result = consensus_method(normalized_df, rescoring_columns, id_column)
    
    # Handle duplicates if needed
    if aggregation == "best":
        result = result.sort_values(result.columns[-1], ascending=False)
        result = result.drop_duplicates(subset=[id_column], keep='first')
    elif aggregation == "avg":
        score_column = result.columns[-1]
        result = result.groupby(id_column)[score_column].mean().reset_index()
    
    # Normalize final consensus scores if requested
    if normalize and len(result) > 0:
        score_column = result.columns[-1]
        scores = result[score_column].values
        min_score, max_score = np.min(scores), np.max(scores)
        if max_score != min_score:
            result[score_column] = (scores - min_score) / (max_score - min_score)
        else:
            result[score_column] = np.zeros_like(scores)
    
    # Save results if output path provided
    if output and not result.empty:
        output_path = Path(output)
        result.to_csv(output_path, index=False)
        return output_path
    
    return result

def save_results(result: pd.DataFrame, output: str | Path) -> Path:
    """Save the results to the specified output file."""
    output_path = Path(output)
    result.to_csv(output_path, index=False)
    return output_path
