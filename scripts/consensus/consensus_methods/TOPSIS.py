import warnings
import pandas as pd
import numpy as np
import skcriteria as skc
from skcriteria.pipeline import mkpipe
from skcriteria.agg import similarity

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def TOPSIS_best(df: pd.DataFrame, selected_columns: list, normalize: bool = True) -> pd.DataFrame:
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
	pipe = mkpipe(similarity.TOPSIS())
	# Determine the weighted sum model rankings
	rankings = pipe.evaluate(dm)
	df["TOPSIS"] = rankings.values
	# Sort the dataframe by the weighted sum model rankings in ascending order
	df = df.sort_values("TOPSIS", ascending=True)
	# Drop duplicate rows based on ID, keeping only the lowest mean rank
	df = df.drop_duplicates(subset="ID", inplace=False)
	# Normalize the TOPSIS column if required
	if normalize:
		df["TOPSIS"] = (df["TOPSIS"].max() - df["TOPSIS"]) / (df["TOPSIS"].max() - df["TOPSIS"].min())
	df = df.rename(columns={"TOPSIS": "TOPSIS_best"})
	return df[["ID", "TOPSIS_best"]]

def TOPSIS_avg(df: pd.DataFrame, selected_columns: list, normalize: bool = True) -> pd.DataFrame:
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
	pipe = mkpipe(similarity.TOPSIS())
	# Determine the weighted sum model rankings
	rankings = pipe.evaluate(dm)
	df["TOPSIS"] = rankings.values
	# Group the dataframe by 'ID' and calculate the mean of numeric columns
	df = df.groupby("ID", as_index=False).mean(numeric_only=True)
	# Normalize the TOPSIS column if required
	if normalize:
		df["TOPSIS"] = (df["TOPSIS"].max() - df["TOPSIS"]) / (df["TOPSIS"].max() - df["TOPSIS"].min())
	df = df.rename(columns={"TOPSIS": "TOPSIS_avg"})
	return df[["ID", "TOPSIS_avg"]]