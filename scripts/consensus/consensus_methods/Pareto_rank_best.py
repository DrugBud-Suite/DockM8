import warnings

import pandas as pd
import numpy as np
from typing import List, Tuple

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def is_pareto_efficient(scores: np.ndarray) -> np.ndarray:
	"""
    Find the pareto-efficient points
    :param scores: An (n_points, n_scores) array
    :return: A boolean array of pareto-efficient points
    """
	is_efficient = np.ones(scores.shape[0], dtype=bool)
	for i, c in enumerate(scores):
		if is_efficient[i]:
			is_efficient[is_efficient] = np.any(scores[is_efficient] > c, axis=1)
			is_efficient[i] = True
	return is_efficient


def Pareto_rank_best(df: pd.DataFrame, selected_columns: list, normalize: bool = True) -> pd.DataFrame:
	"""
    Compute the Pareto rank for each compound
    :param df: DataFrame containing the compounds and their scores
    :param selected_columns: List of column names containing the scores
    :return: DataFrame with added 'Pareto_Rank' column
    """
	df = df[["ID"] + selected_columns]
	df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors="coerce")
	scores = df[selected_columns].values
	remaining = np.ones(len(df), dtype=bool)
	ranks = np.zeros(len(df), dtype=int)
	rank = 1

	while np.any(remaining):
		pareto_efficient = is_pareto_efficient(scores[remaining])
		ranks[remaining] = np.where(pareto_efficient, rank, ranks[remaining])
		remaining[remaining] = ~pareto_efficient
		rank += 1

	df['Pareto_rank'] = ranks
	df = df[['ID', 'Pareto_rank']]
	df = df.sort_values("Pareto_rank", ascending=False)
	df = df.drop_duplicates("ID", inplace=False)
	if normalize:
		df['Pareto_rank'] = (df['Pareto_rank'].max() - df['Pareto_rank']) / (df['Pareto_rank'].max() -
																				df['Pareto_rank'].min())
	df = df.rename(columns={'Pareto_rank': 'Pareto_rank_best'})
	return df[['ID', 'Pareto_rank_best']]
