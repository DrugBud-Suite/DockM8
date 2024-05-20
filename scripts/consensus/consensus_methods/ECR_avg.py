import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def ECR_avg(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
	"""
    Calculates the Exponential Consensus Ranking (ECR) score for each ID in the input dataframe. Averaging of the score is done across all selected poses.

    Args:
        df (pd.DataFrame): Input dataframe containing the data.
        clustering_metric (str): Clustering metric to be used in the ECR calculation.
        selected_columns (list): List of column names to be used for ECR calculation.

    Returns:
        pd.DataFrame: Dataframe with two columns: 'ID' and `Method2_ECR_{clustering_metric}`. Each row represents an ID and its corresponding average ECR rank across the clustered poses.
    """
	# Select the 'ID' column and the selected columns from the input dataframe
	df = df[["ID"] + selected_columns]
	# Convert selected columns to numeric values
	df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")
	# Calculate the sigma value
	sigma = 0.05 * len(df)
	# Calculate the ECR values for each selected column
	ecr_scores = np.exp(-df[selected_columns] / sigma)
	# Sum the ECR values across each row
	df["ECR"] = ecr_scores[selected_columns].sum(axis=1) / sigma
	# Drop the selected columns from the dataframe
	df = df[["ID", "ECR"]]
	# Group the dataframe by 'ID' and calculate the mean of numeric columns
	df = df.groupby("ID", as_index=False).mean(numeric_only=True)
	# Normalize the ECR column
	df["ECR"] = (df["ECR"] - df["ECR"].min()) / (df["ECR"].max() - df["ECR"].min())
	df = df.rename(columns={"ECR": f"ECR_avg_{clustering_metric}"})
	# Return the dataframe with 'ID' and 'ECR_avg_{clustering_metric}' columns
	return df[["ID", f"ECR_avg_{clustering_metric}"]]
