import warnings

import pandas as pd

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def Zscore_best(df: pd.DataFrame, selected_columns: list, normalize: bool = True) -> pd.DataFrame:
	"""
    Calculates the Z-score consensus scores for each row in the given DataFrame. Returns only the best score for each ID. No averaging is carried out.

    Args:
    - df: A pandas DataFrame containing the input data with columns 'ID', 'Score1', 'Score2', and so on.
    - clustering_metric: A string representing the clustering metric used.
    - selected_columns: A list of strings representing the selected columns for calculating the Z-score score.

    Returns:
    - A pandas DataFrame with columns 'ID' and 'Zscore_best_{clustering_metric}', where 'Zscore_best_{clustering_metric}' represents the Z-score score for each ID. Only the row with the highest Z-score for each ID is included in the output.
    """
	# Select the 'ID' column and the selected columns from the input dataframe
	df = df[["ID"] + selected_columns]
	# Convert selected columns to numeric values
	df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")
	# Calculate Z-scores
	z_scores = (df[selected_columns] - df[selected_columns].mean()) / df[selected_columns].std()
	# Calculate mean Z-score for each row
	df["Zscore"] = z_scores.mean(axis=1)
	df = df[["ID", "Zscore"]]
	# Sort the dataframe by Z-scores in descending order
	df = df.sort_values("Zscore", ascending=False)
	# Drop duplicate rows based on ID, keeping only the highest Z-score
	df = df.drop_duplicates("ID", inplace=False)
	# Normalize the Zscore column
	if normalize:
		df["Zscore"] = (df["Zscore"] - df["Zscore"].min()) / (df["Zscore"].max() - df["Zscore"].min())
	df = df.rename(columns={"Zscore": "Zscore_best"})
	return df[["ID", "Zscore_best"]]
