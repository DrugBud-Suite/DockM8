import warnings

import pandas as pd

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def RbR_avg(df: pd.DataFrame, selected_columns: list, normalize: bool = True) -> pd.DataFrame:
	"""
    Calculates the Rank by Rank (RbR) consensus score for each ID in the input dataframe. Averaging of the score is done across all selected poses.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the input data with columns 'ID', 'Score1', 'Score2', and so on.
        clustering_metric (str): A string representing the clustering metric used.
        selected_columns (list): A list of strings representing the selected columns for calculating the RbR score.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns 'ID' and 'RbR_{clustering_metric}', where 'RbR_{clustering_metric}' represents the RbR score for each ID.
    """
	# Select the 'ID' column and the selected columns from the input dataframe
	df = df[['ID'] + selected_columns]
	# Convert selected columns to numeric values
	df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors='coerce')
	# Calculate the mean rank across the selected columns
	df['RbR'] = df[selected_columns].mean(axis=1)
	df = df[['ID', 'RbR']]
	# Group the dataframe by 'ID' and calculate the mean of numeric columns
	df = df.groupby('ID', as_index=False).mean(numeric_only=True)
	# Normalize the RbR column
	if normalize:
		df['RbR'] = (df['RbR'].max() - df['RbR']) / (df['RbR'].max() - df['RbR'].min())
	df = df.rename(columns={'RbR': 'RbR_avg'})
	return df[['ID', 'RbR_avg']]
