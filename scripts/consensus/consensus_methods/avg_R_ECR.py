import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def avg_R_ECR(df: pd.DataFrame, selected_columns: list, normalize: bool = True) -> pd.DataFrame:
	"""
    Averages the ranks across poses, reranks them then calculates the Exponential Consensus Ranking (ECR) for a given dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing the data.
        clustering_metric (str): The name of the clustering metric.
        selected_columns (list): The list of selected columns to calculate ECR for.

    Returns:
        pd.DataFrame: The output dataframe with columns 'ID' and 'avg_ECR_clustering' representing the ID and Exponential Consensus Ranking values for the selected columns.
    """
	# Select the 'ID' column and the selected columns from the input dataframe
	df = df[["ID"] + selected_columns]
	# Convert selected columns to numeric values
	df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")
	# Calculate the mean ranks for the selected columns
	df = df.groupby("ID", as_index=False).mean(numeric_only=True).round(2)
	# Rerank the mean ranks
	df[selected_columns] = df[selected_columns].rank(method="average", ascending=True, numeric_only=True)
	# Calculate the sigma value
	sigma = 0.05 * len(df)
	# Calculate the ECR values using the formula
	ecr_values = np.exp(-df[selected_columns] / sigma)
	# Sum the ECR values across each row
	df["ECR"] = ecr_values[selected_columns].sum(axis=1) / sigma
	# Normalize the ECR column
	if normalize:
		df["ECR"] = (df["ECR"] - df["ECR"].min()) / (df["ECR"].max() - df["ECR"].min())
	df = df.rename(columns={"ECR": "avg_R_ECR"})
	return df[["ID", "avg_R_ECR"]]
