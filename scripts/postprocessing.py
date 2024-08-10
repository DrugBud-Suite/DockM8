import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.consensus.consensus import CONSENSUS_METHODS
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS
from scripts.utilities.logging import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def standardize_scores(df: pd.DataFrame, standardization_type: str):
	"""
	Standardizes the scores in the given dataframe.

	Args:
	- df: pandas dataframe containing the scores to be standardized
	- standardization_type: string indicating the type of standardization ('min_max', 'scaled', 'percentiles')

	Returns:
	- df: pandas dataframe with standardized scores
	"""

	def min_max_standardization(score, best_value, min_value, max_value):
		"""
		Performs min-max standardization scaling on a given score using the defined min and max values.
		"""
		if best_value == "max":
			return (score-min_value) / (max_value-min_value)
		else:  # best_value == "min"
			return (max_value-score) / (max_value-min_value)

	for col in df.columns:
		if col not in ["Pose ID", "ID", 'SMILES', 'Molecule']:
			# Convert column to numeric values
			df[col] = pd.to_numeric(df[col], errors="coerce")
			# Get information about the scoring function
			function_info = RESCORING_FUNCTIONS.get(col)
			if function_info:
				if standardization_type == "min_max":
					# Standardise using the score's (current distribution) min and max values
					df[col] = min_max_standardization(df[col],
														function_info["best_value"],
														df[col].min(),
														df[col].max())
				elif standardization_type == "scaled":
					# Standardise using the range defined in the RESCORING_FUNCTIONS dictionary
					if function_info["best_value"] == "max":
						df[col] = min_max_standardization(df[col],
															function_info["best_value"],
															function_info["score_range"][0],
															function_info["score_range"][1])
					else:
						df[col] = min_max_standardization(df[col],
															function_info["best_value"],
															function_info["score_range"][1],
															function_info["score_range"][0])
				elif standardization_type == "percentiles":
					# Standardise using the 1st and 99th percentiles of this distribution
					column_data = df[col].dropna().values
					col_min, col_max = np.percentile(column_data, [1, 99])
					df[col] = min_max_standardization(df[col], function_info["best_value"], col_min, col_max)
				else:
					raise ValueError(f"Invalid standardization type: {standardization_type}")
			else:
				print(f"Warning: No function info found for column {col}. Skipping standardization.")

	return df


def rank_scores(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Ranks the scores in the given DataFrame in descending order for each column, excluding 'Pose ID' and 'ID'.

	Parameters:
		df (pandas.DataFrame): The DataFrame containing the scores to be ranked.

	Returns:
		pandas.DataFrame: The DataFrame with the scores ranked in descending order for each column.
	"""
	df = df.assign(
		**{
			col: df[col].rank(method="average", ascending=False)
			for col in df.columns
			if col not in ["Pose ID", "ID", 'SMILES', 'Molecule']})
	return df


import pandas as pd
from rdkit.Chem import PandasTools
from pathlib import Path


def apply_consensus_methods(input_file: Path, consensus_methods: str, standardization_type: str):
	"""
	Applies consensus methods to rescored data from a CSV or SDF file and saves the results.

	Args:
	input_file (Path): Path to the input file (CSV or SDF) containing rescored data.
	consensus_methods (str): The consensus methods to apply.
	standardization_type (str): The type of standardization to apply to the scores.

	Returns:
	None
	"""
	if consensus_methods is None or consensus_methods == "None":
		return print("No consensus methods selected, skipping consensus.")

	print(f"Applying consensus methods: {consensus_methods}")

	# Determine file type and read accordingly
	file_extension = input_file.suffix.lower()
	if file_extension == '.csv':
		rescored_dataframe = pd.read_csv(input_file)
		if 'SMILES' in rescored_dataframe.columns:
			rescored_dataframe['Molecule'] = rescored_dataframe['SMILES'].apply(lambda mol: Chem.MolFromSmiles(mol)
																				if mol is not None else None)
		else:
			pass
	elif file_extension == '.sdf':
		rescored_dataframe = PandasTools.LoadSDF(input_file,
													molColName="Molecule",
													idName="Pose ID",
													smilesName="SMILES")
	else:
		raise ValueError(f"Unsupported file type: {file_extension}. Please provide a CSV or SDF file.")

	# Standardize the scores and add the 'ID' column
	standardized_dataframe = standardize_scores(rescored_dataframe, standardization_type)

	# Rank the scores and add the 'ID' column
	ranked_dataframe = rank_scores(standardized_dataframe)

	# Ensure consensus_methods is a list even if it's a single string
	if isinstance(consensus_methods, str):
		consensus_methods = [consensus_methods]

	output_dir = Path(input_file).parent / "consensus"
	output_dir.mkdir(parents=True, exist_ok=True)

	for consensus_method in consensus_methods:
		if consensus_method not in CONSENSUS_METHODS:
			raise ValueError(f"Invalid consensus method: {consensus_method}")

		consensus_info = CONSENSUS_METHODS[consensus_method]
		consensus_type = consensus_info["type"]
		consensus_function = consensus_info["function"]

		if consensus_type == "rank":
			consensus_dataframe = consensus_function(
				ranked_dataframe,
				[col for col in ranked_dataframe.columns if col not in ["Pose ID", "ID", "Molecule", "SMILES"]])
		elif consensus_type == "score":
			consensus_dataframe = consensus_function(
				standardized_dataframe,
				[col for col in standardized_dataframe.columns if col not in ["Pose ID", "ID", "Molecule", "SMILES"]])
		else:
			raise ValueError(f"Invalid consensus method type: {consensus_type}")

		consensus_dataframe = consensus_dataframe.drop(columns="Pose ID", errors="ignore")
		consensus_dataframe = consensus_dataframe.sort_values(by=consensus_method, ascending=False)

		# Save results
		if 'Molecule' in rescored_dataframe.columns:
			consensus_dataframe = pd.merge(consensus_dataframe,
											rescored_dataframe[["ID", "Molecule", "SMILES"]],
											on="ID",
											how="left")
			PandasTools.WriteSDF(consensus_dataframe,
									str(input_file.parent /
										f"consensus_results_{consensus_method}_{standardization_type}.sdf"),
									molColName="Molecule",
									idName="ID",
									properties=list(consensus_dataframe.columns))
		else:
			consensus_dataframe = pd.merge(consensus_dataframe,
											rescored_dataframe[["ID", "SMILES"]],
											on="ID",
											how="left")
		consensus_dataframe = consensus_dataframe.drop(columns="Molecule", errors="ignore")
		consensus_dataframe.to_csv(input_file.parent /
									f"consensus_results_{consensus_method}_{standardization_type}.csv",
									index=False)

		return consensus_dataframe


# Note: The CONSENSUS_METHODS, standardize_scores, and rank_scores functions/variables
# are assumed to be defined elsewhere in your code.


def ensemble_consensus(receptors: list, selection_method: str, consensus_method: str, threshold: float):
	"""
	Given a list of receptor file paths, this function reads the consensus clustering results for each receptor,
	selects the top n compounds based on a given threshold, and returns a list of common compounds across all receptors.

	Parameters:
	-----------
	receptors : list of str
		List of file paths to receptor files.
	selection_method : str
		The clustering metric used to generate the consensus clustering results.
	consensus_method : str
		The clustering method used to generate the consensus clustering results.
	threshold : float or int
		The percentage of top compounds to select from each consensus clustering result.

	Returns:
	--------
	list of str
		List of common compounds across all receptors.
	"""
	topn_dataframes = []
	# Iterate over each receptor file
	for receptor in receptors:
		w_dir = Path(receptor).parent / Path(receptor).stem
		# Read the consensus clustering results for the receptor
		if selection_method in [
			"bestpose_GNINA", "bestpose_SMINA", "bestpose_PLANTS", "bestpose_QVINAW", "bestpose_QVINA2", ] + list(
			RESCORING_FUNCTIONS.keys()):
			consensus_file = PandasTools.LoadSDF(str(w_dir / "consensus" /
				f"{selection_method}_{consensus_method}_results.sdf"),
				molColName="Molecule",
				idName="ID")

		else:
			consensus_file = pd.read_csv(
				Path(w_dir) / "consensus" / f"{selection_method}_{consensus_method}_results.csv")
		# Select the top n compounds based on the given threshold
		consensus_file_topn = consensus_file.head(math.ceil(consensus_file.shape[0] * (threshold/100)))
		# Append the top n compounds dataframe to the list
		topn_dataframes.append(consensus_file_topn)
	# Find the common compounds across all receptors
	common_compounds = set(topn_dataframes[0]["ID"])
	# Find the intersection of 'ID' values with other dataframes
	for df in topn_dataframes[1:]:
		common_compounds.intersection_update(df["ID"])
	common_compounds_list = list(common_compounds)

	common_compounds_df = pd.DataFrame()

	for receptor in receptors:
		w_dir = Path(receptor).parent / Path(receptor).stem
		# Read the consensus clustering results for the receptor
		if selection_method in [
			"bestpose_GNINA", "bestpose_SMINA", "bestpose_PLANTS", "bestpose_QVINAW", "bestpose_QVINA2", ] + list(
			RESCORING_FUNCTIONS.keys()):
			consensus_file = PandasTools.LoadSDF(str(w_dir / "consensus" /
				f"{selection_method}_{consensus_method}_results.sdf"),
				molColName="Molecule",
				idName="ID")

		else:
			consensus_file = pd.read_csv(
				Path(w_dir) / "consensus" / f"{selection_method}_{consensus_method}_results.csv")
		consensus_file = consensus_file[consensus_file["ID"].isin(common_compounds_list)]
		consensus_file["Receptor"] = Path(receptor).stem
		common_compounds_df = pd.concat([common_compounds_df, consensus_file], axis=0)
	# Save the common compounds and CSV or SDF file
	if selection_method in [
		"bestpose_GNINA", "bestpose_SMINA", "bestpose_PLANTS", "bestpose_QVINAW", "bestpose_QVINA2", ] + list(
		RESCORING_FUNCTIONS.keys()):
		PandasTools.WriteSDF(common_compounds_df,
			str(Path(receptors[0]).parent / "ensemble_results.sdf"),
			molColName="Molecule",
			idName="ID",
			properties=list(common_compounds_df.columns))

	else:
		common_compounds_df.to_csv(Path(receptors[0]).parent / "ensemble_results.csv", index=False)
	return common_compounds_df
