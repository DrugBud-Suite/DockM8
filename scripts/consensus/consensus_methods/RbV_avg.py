import warnings

import pandas as pd

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def RbV_avg(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
	"""
    Calculates the Rank by Vote consensus for a given DataFrame. Averaging of the score is done across all selected poses.

    Args:
        df (pd.DataFrame): A DataFrame containing the data.
        clustering_metric (str): A string representing the clustering metric.
        selected_columns (list): A list of column names to consider for the calculation.

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'ID' and 'RbV_' followed by the 'clustering_metric' value.
                     The 'RbV_' column contains the Rank by Vote consensus scores for each ID.
    """
	# Select the 'ID' column and the selected columns from the input dataframe
	df = df[["ID"] + selected_columns]
	# Convert selected columns to numeric values
	df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")
	# Initialize a new column 'vote' in the DataFrame with default value 0
	df["RbV"] = 0
	# Increment the f'RbV_{clustering_metric}' column by 1 if the value in each selected column is greater than the 95th percentile of the column
	for column in selected_columns:
		df["RbV"] += (df[column] > df[column].quantile(0.95)).astype(int)
	df = df[["ID", "RbV"]]
	# Group the DataFrame by 'ID' and average across poses
	df = df.groupby("ID", as_index=False).mean(numeric_only=True).round(2)
	# Normalize the RbV column
	df["RbV"] = (df["RbV"] - df["RbV"].min()) / (df["RbV"].max() - df["RbV"].min())
	df = df.rename(columns={"RbV": f"RbV_avg_{clustering_metric}"})
	# Return the DataFrame with columns 'ID' and 'RbV_' followed by the clustering_metric value
	return df[["ID", f"RbV_avg_{clustering_metric}"]]