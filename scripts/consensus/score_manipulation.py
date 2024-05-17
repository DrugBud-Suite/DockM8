from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd


# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts"
                     for p in Path(__file__).resolve().parents
                     if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring import RESCORING_FUNCTIONS


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def standardize_scores(df: pd.DataFrame, standardization_type: str):
    """
    Standardizes the scores in the given dataframe.

    Args:
    - df: pandas dataframe containing the scores to be standardized
    - standardization_type: string indicating the type of standardization ('min_max', 'scaled', 'percentiles')

    Returns:
    - df: pandas dataframe with standardized scores
    """

    def min_max_standardization(score, best_value, min_value, max_value):
        """
        Performs min-max standardization scaling on a given score using the defined min and max values.
        """
        return ((score - min_value) /
                (max_value - min_value) if best_value == "max" else
                (max_value - score) / (max_value - min_value))

    for col in df.columns:
        if col not in ["Pose ID", "ID"]:
            # Convert column to numeric values
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Get information about the scoring function
            function_info = RESCORING_FUNCTIONS.get(col)
            if function_info:
                if standardization_type == "min_max":
                    # Standardise using the score's (current distribution) min and max values
                    df[col] = min_max_standardization(
                        df[col], function_info["best_value"], df[col].min(),
                        df[col].max())
                elif standardization_type == "scaled":
                    # Standardise using the range defined in the RESCORING_FUNCTIONS dictionary
                    df[col] = min_max_standardization(
                        df[col], function_info["best_value"],
                        *function_info["range"])
                elif standardization_type == "percentiles":
                    # Standardise using the 1st and 99th percentiles of this distribution
                    column_data = df[col].dropna().values
                    col_min, col_max = (np.percentile(column_data, [
                        1, 99]) if function_info["best_value"] == "max" else
                                        np.percentile(column_data, [99, 1]))
                    df[col] = min_max_standardization(
                        df[col], function_info["best_value"], col_min, col_max)
                else:
                    raise ValueError(
                        f"Invalid standardization type: {standardization_type}")

    return df


def rank_scores(input_dataframe: pd.DataFrame):
    """
    Ranks the scores in the given DataFrame in descending order for each column, excluding 'Pose ID' and 'ID'.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the scores to be ranked.

    Returns:
        pandas.DataFrame: The DataFrame with the scores ranked in descending order for each column.
    """
    ranked_dataframe = input_dataframe.assign(
        **{
            col: input_dataframe[col].rank(method="average", ascending=False)
            for col in input_dataframe.columns
            if col not in ["Pose ID", "ID"]})
    return ranked_dataframe