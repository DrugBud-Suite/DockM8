import warnings

import pandas as pd
import numpy as np
from typing import List, Tuple

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


import numba


def is_pareto_efficient(costs, return_mask=True):
	"""
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
	is_efficient = np.arange(costs.shape[0])
	n_points = costs.shape[0]
	next_point_index = 0                                                # Next index in the is_efficient array to search for
	while next_point_index < len(costs):
		nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
		nondominated_point_mask[next_point_index] = True
		is_efficient = is_efficient[nondominated_point_mask]               # Remove dominated points
		costs = costs[nondominated_point_mask]
		next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
	if return_mask:
		is_efficient_mask = np.zeros(n_points, dtype=bool)
		is_efficient_mask[is_efficient] = True
		return is_efficient_mask
	else:
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
