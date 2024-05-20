import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def ECR_best(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
	"""
    Calculates the Exponential Consensus Ranking (ECR) score for each ID in the rescored dataframe. Returns the highest ECR score for each ID.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the rescored data with columns 'ID', 'Score1', 'Score2', and so on.
        clustering_metric (str): A string representing the clustering metric used.
        selected_columns (list): A list of strings representing the selected columns for calculating the ECR score.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns 'ID' and 'Method1_ECR_{clustering_metric}', where 'Method1_ECR_{clustering_metric}' represents the ECR score for each ID.
    """
	# Select the 'ID' column and the selected columns from the input dataframe
	df = df[["ID"] + selected_columns]
	# Convert selected columns to numeric values
	df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")
	# Calculate the sigma value
	sigma = 0.05 * len(df)
	# Calculate ECR scores for each value in selected columns
	ecr_scores = np.exp(-(df[selected_columns] / sigma))
	# Sum the ECR scores for each ID
	df["ECR"] = ecr_scores[selected_columns].sum(axis=1) / sigma
	# Drop the selected columns
	df = df[["ID", "ECR"]]
	# Sort by ECR scores in descending order
	df.sort_values("ECR", ascending=False, inplace=True)
	# Drop duplicate rows based on ID, keeping only the highest ECR score
	df.drop_duplicates(subset="ID", inplace=True)
	# Normalize the ECR column
	df["ECR"] = (df["ECR"] - df["ECR"].min()) / (df["ECR"].max() - df["ECR"].min())
	df = df.rename(columns={"ECR": f"ECR_best_{clustering_metric}"})
	# Return dataframe with ID and ECR scores
	return df[["ID", f"ECR_best_{clustering_metric}"]]
