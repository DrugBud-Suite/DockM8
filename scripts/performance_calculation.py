import itertools
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit.ML.Scoring import Scoring
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import skcriteria as skc
from skcriteria.agg import simple

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.consensus.consensus import CONSENSUS_METHODS
from scripts.consensus.score_manipulation import rank_scores, standardize_scores
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.results import DockM8Results

warnings.filterwarnings("ignore")


def process_combination(combination,
						selection_method,
						ranked_df,
						standardised_df,
						actives_df,
						percentages,
						docking_program='None'):
	filtered_ranked_df = ranked_df[["ID"] + list(combination)]
	filtered_standardised_df = standardised_df[["ID"] + list(combination)]
	combination_dfs = []
	consensus_dfs = {}

	# For each consensus method
	for method in CONSENSUS_METHODS.keys():
		if CONSENSUS_METHODS[method]["type"] == "rank":
			consensus_dfs[method] = CONSENSUS_METHODS[method]["function"](filtered_ranked_df, list(combination))
		elif CONSENSUS_METHODS[method]["type"] == "score":
			consensus_dfs[method] = CONSENSUS_METHODS[method]["function"](filtered_standardised_df, list(combination))

		merged_df = pd.merge(consensus_dfs[method], actives_df, on="ID")
		# Get the column name that is not 'ID' or 'Activity'
		col_to_sort = [col for col in merged_df.columns if col not in ["ID", "Activity"]][0]
		merged_df = merged_df.sort_values(col_to_sort, ascending=False)
		merged_df = merged_df.fillna(0)
		scores = merged_df[col_to_sort].values
		activities = merged_df["Activity"].values

		# Calculate performance metrics
		auc_roc = round(roc_auc_score(activities, scores, multi_class="ovo"), 3)
		bedroc = round(Scoring.CalcBEDROC(list(zip(scores, activities)), 1, 80.5), 3)
		auc = round(Scoring.CalcAUC(list(zip(scores, activities)), 1), 3)
		ef_results = [calculate_EF(merged_df, p) for p in percentages]
		rie = round(Scoring.CalcRIE(list(zip(scores, activities)), 1, 80.5), 3)

		combination_dfs.append(
			pd.DataFrame(
				{
					"docking_program": docking_program,
					"selection_method": selection_method,
					"consensus": method,
					"scoring": "_".join(list(combination)),
					"AUC_ROC": auc_roc,
					"BEDROC": bedroc,
					"AUC": auc,
					**{
						f"EF_{p}%": ef for p, ef in zip(percentages, ef_results)},
					"RIE": rie, },
				index=[0],
			))

	combination_df = pd.concat(combination_dfs, axis=0)
	return combination_df


def calculate_performance(w_dir: Path, results: DockM8Results, actives_library: Path, percentages: list):
	printlog("Calculating performance...")

	# Load actives data
	actives_df = PandasTools.LoadSDF(str(actives_library), molColName=None, idName="ID")
	actives_df = actives_df[["ID", "Activity"]]
	actives_df["Activity"] = pd.to_numeric(actives_df["Activity"])

	all_results = []

	for selection_method, rescored_poses in results.rescored_poses.items():
		# Use join() to combine multiple docking programs if there are more than one
		docking_programs = ", ".join(results.docking_programs)
		printlog(f"Processing {docking_programs} with {selection_method} selection method...")
		rescored_dataframe = pd.read_csv(rescored_poses)
		performance_results = calculate_performance_for_method(rescored_dataframe,
																actives_df,
																percentages,
																docking_programs,
																selection_method)

		all_results.append(performance_results)

	combined_results = pd.concat(all_results, ignore_index=True)

	# Save results to CSV
	output_file = w_dir / "performance.csv"
	combined_results.to_csv(output_file, index=False)
	printlog(f"Performance results saved to {output_file}")

	return combined_results


# The calculate_EF function remains unchanged
def calculate_EF(merged_df, percentage: float):
	total_rows = len(merged_df)
	N100_percent = total_rows

	Nx_percent = round((percentage/100) * total_rows)
	Hits100_percent = np.sum(merged_df["Activity"])

	Hitsx_percent = np.sum(merged_df.head(Nx_percent)["Activity"])

	ef = (Hitsx_percent/Nx_percent) * (N100_percent/Hits100_percent)
	if ef > 100:
		return 100
	else:
		return round(ef, 2)


def calculate_performance_for_method(rescored_df, actives_df, percentages, docking_program, selection_method):
	standardised_df = standardize_scores(rescored_df, "min_max")
	standardised_df["ID"] = standardised_df["Pose ID"].str.split("_").str[0]
	standardised_df["ID"] = standardised_df["ID"].astype(str)
	score_columns = [col for col in standardised_df.columns if col not in ["Pose ID", "ID", "Molecule", "SMILES"]]

	result_list = []
	merged_df = pd.merge(standardised_df, actives_df, on="ID")
	merged_df.fillna(0, inplace=True)

	# Calculate performance for single scoring functions
	for col in score_columns:
		merged_df.sort_values(col, ascending=False, inplace=True)
		scores = merged_df[col].values
		activities = merged_df["Activity"].values

		auc_roc = round(roc_auc_score(activities, scores, multi_class="ovo"), 3)
		bedroc = round(Scoring.CalcBEDROC(list(zip(scores, activities)), 1, 80.5), 3)
		auc = round(Scoring.CalcAUC(list(zip(scores, activities)), 1), 3)
		ef_results = [calculate_EF(merged_df, p) for p in percentages]
		rie = round(Scoring.CalcRIE(list(zip(scores, activities)), 1, 80.5), 3)

		result_list.append(
			pd.DataFrame(
				{
					"docking_program": docking_program,
					"selection_method": selection_method,
					"consensus": "None",
					"scoring": col,
					"AUC_ROC": auc_roc,
					"BEDROC": bedroc,
					"AUC": auc,
					**{
						f"EF_{p}%": ef for p, ef in zip(percentages, ef_results)},
					"RIE": rie, },
				index=[0],
			))

	# Calculate performance for consensus scoring functions
	ranked_df = rank_scores(standardised_df)
	ranked_df["ID"] = ranked_df["Pose ID"].str.split("_").str[0]
	ranked_df["ID"] = ranked_df["ID"].astype(str)

	for length in tqdm(range(2, len(score_columns) + 1), desc=f"{docking_program}_{selection_method}"):
		combinations = list(itertools.combinations(score_columns, length))
		results = parallel_executor(process_combination,
									combinations,
									n_cpus=32,
									job_manager="concurrent_process_silent",
									docking_program=docking_program,
									selection_method=selection_method,
									ranked_df=ranked_df,
									standardised_df=standardised_df,
									actives_df=actives_df,
									percentages=percentages)

		for result in results:
			result_list.append(result)

	return pd.concat(result_list, axis=0)


def determine_optimal_conditions(w_dir: Path, results: DockM8Results, actives_library: Path, percentages: list):
	printlog("Determining optimal conditions using the actives library : " + str(actives_library))

	# Load actives data
	actives_df = PandasTools.LoadSDF(str(actives_library), molColName=None, idName="ID")
	actives_df = actives_df[["ID", "Activity"]]
	actives_df["Activity"] = pd.to_numeric(actives_df["Activity"])

	all_results = []

	for selection_method, rescored_poses in results.rescored_poses.items():
		printlog(f"Processing {results.docking_program} with {selection_method} selection method...")
		rescored_dataframe = pd.read_csv(rescored_poses)
		performance_results = calculate_performance_for_method(rescored_dataframe,
																actives_df,
																percentages,
																results.docking_program,
																selection_method)

		all_results.append(performance_results)

	combined_results = pd.concat(all_results, ignore_index=True)

	# Save results to CSV
	output_file = w_dir / "optimal_conditions.csv"
	combined_results.to_csv(output_file, index=False)
	printlog(f"Decoy performance results saved to {output_file}")

	metrics = ['AUC_ROC', 'BEDROC', 'AUC', 'EF_0.5%', 'EF_1%', 'EF_2%', 'EF_5%', 'EF_10%', 'RIE']
	matrix = combined_results[metrics].values

	objectives = ["max"] * len(metrics)             # All metrics should be maximized

	# Create the decision matrix
	dm = skc.mkdm(matrix, objectives, alternatives=combined_results.index.tolist(), criteria=metrics, )
	dec = simple.WeightedSumModel()

	# Get the rankings
	rankings = dec.evaluate(dm)

	combined_results['Method_rank'] = rankings.values

	combined_results = combined_results.sort_values('Method_rank', ascending=True)

	optimal_method = combined_results.iloc[0]

	return optimal_method.to_dict()
