import warnings
import pandas as pd
import numpy as np
import skcriteria as skc
from skcriteria.agg import simple

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def WeightedSumModel_best(df: pd.DataFrame, selected_columns: list, normalize: bool = True) -> pd.DataFrame:
	"""
	Calculates the Sum Model consensus scores for each row in the given DataFrame.
	All criteria are treated as maximization criteria.

	Args:
	- df: A pandas DataFrame containing the input data with columns 'ID', 'Score1', 'Score2', and so on.
	- selected_columns: A list of strings representing the selected columns for calculating the Sum score.
	- normalize: A boolean indicating whether to normalize the final scores.

	Returns:
	- A pandas DataFrame with columns 'ID' and 'SumScore', where 'SumScore' represents the Sum score for each ID.
	"""
	# Select the 'ID' column and the selected columns from the input dataframe
	df = df[["ID"] + selected_columns]

	# Create the decision matrix
	dm = skc.mkdm(
		df[selected_columns].values,
		["max"] * len(selected_columns),
		alternatives=df['ID'].values.tolist(),
		criteria=selected_columns)
	# Calculate the weighted sum model
	dec = simple.WeightedSumModel()
	# Determine the weighted sum model rankings
	rankings = dec.evaluate(dm)
	df["WeightedSumModel"] = rankings.values
	# Sort the dataframe by the weighted sum model rankings in ascending order
	df = df.sort_values("WeightedSumModel", ascending=True)
	# Drop duplicate rows based on ID, keeping only the lowest mean rank
	df = df.drop_duplicates(subset="ID", inplace=False)
	# Normalize the WeightedSumModel column if required
	if normalize:
		df["WeightedSumModel"] = (df["WeightedSumModel"].max() - df["WeightedSumModel"]) / (df["WeightedSumModel"].max() - df["WeightedSumModel"].min())
	df = df.rename(columns={"WeightedSumModel": "WeightedSumModel_best"})
	return df[["ID", "WeightedSumModel_best"]]

def WeightedSumModel_avg(df: pd.DataFrame, selected_columns: list, normalize: bool = True) -> pd.DataFrame:
	"""
	Calculates the Sum Model consensus scores for each row in the given DataFrame.
	All criteria are treated as maximization criteria.

	Args:
	- df: A pandas DataFrame containing the input data with columns 'ID', 'Score1', 'Score2', and so on.
	- selected_columns: A list of strings representing the selected columns for calculating the Sum score.
	- normalize: A boolean indicating whether to normalize the final scores.

	Returns:
	- A pandas DataFrame with columns 'ID' and 'SumScore', where 'SumScore' represents the Sum score for each ID.
	"""
	# Select the 'ID' column and the selected columns from the input dataframe
	df = df[["ID"] + selected_columns]
	# Create the decision matrix
	dm = skc.mkdm(
		df[selected_columns].values,
		["max"] * len(selected_columns),
		alternatives=df['ID'].values.tolist(),
		criteria=selected_columns)
	# Calculate the weighted sum model
	dec = simple.WeightedSumModel()
	# Determine the weighted sum model rankings
	rankings = dec.evaluate(dm)
	df["WeightedSumModel"] = rankings.values
	# Group the dataframe by 'ID' and calculate the mean of numeric columns
	df = df.groupby("ID", as_index=False).mean(numeric_only=True)
	# Normalize the WeightedSumModel column if required
	if normalize:
		df["WeightedSumModel"] = (df["WeightedSumModel"].max() - df["WeightedSumModel"]) / (df["WeightedSumModel"].max() - df["WeightedSumModel"].min())
	df = df.rename(columns={"WeightedSumModel": "WeightedSumModel_avg"})
	return df[["ID", "WeightedSumModel_avg"]]