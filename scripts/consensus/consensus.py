import math
import sys
import warnings
from pathlib import Path
from typing import Optional, Union, Tuple, List

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.consensus.consensus_methods.avg_ECR import avg_ECR
from scripts.consensus.consensus_methods.avg_R_ECR import avg_R_ECR
from scripts.consensus.consensus_methods.ECR_avg import ECR_avg
from scripts.consensus.consensus_methods.ECR_best import ECR_best
from scripts.consensus.consensus_methods.RbR_avg import RbR_avg
from scripts.consensus.consensus_methods.RbR_best import RbR_best
from scripts.consensus.consensus_methods.RbV_avg import RbV_avg
from scripts.consensus.consensus_methods.RbV_best import RbV_best
from scripts.consensus.consensus_methods.Zscore_avg import Zscore_avg
from scripts.consensus.consensus_methods.Zscore_best import Zscore_best
from scripts.consensus.score_manipulation import rank_scores, standardize_scores
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS
from scripts.utilities.logging import printlog
from scripts.utilities.utilities import parallel_SDF_loader

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

CONSENSUS_METHODS = {
	'avg_ECR': {
		'function': avg_ECR, 'type': 'rank'},
	'avg_R_ECR': {
		'function': avg_R_ECR, 'type': 'rank'},
	'ECR_avg': {
		'function': ECR_avg, 'type': 'rank'},
	'ECR_best': {
		'function': ECR_best, 'type': 'rank'},
	'RbR_avg': {
		'function': RbR_avg, 'type': 'rank'},
	'RbR_best': {
		'function': RbR_best, 'type': 'rank'},
	'RbV_avg': {
		'function': RbV_avg, 'type': 'score'},
	'RbV_best': {
		'function': RbV_best, 'type': 'score'},
	'Zscore_avg': {
		'function': Zscore_avg, 'type': 'score'},
	'Zscore_best': {
		'function': Zscore_best, 'type': 'score'}}


def apply_consensus_methods(poses_input: Union[Path, pd.DataFrame],
							consensus_methods: Union[str, List[str]],
							standardization_type: str,
							output_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""
    Applies consensus methods to rescored data and saves the results to CSV and SDF files.

    Args:
    poses_input: Can be a path to a CSV/SDF file, or a pandas DataFrame containing the poses data.
    consensus_methods: The consensus methods to apply. Can be a single method name or a list of method names.
    standardization_type (str): The type of standardization to apply to the scores.
    output_path (Path, optional): The path where output files should be saved. If None, files won't be saved.

    Returns:
    tuple: (consensus_dataframe_csv, consensus_dataframe_sdf) - DataFrames with consensus results
    """
	# Check if consensus_methods is None or 'None'
	if consensus_methods is None or consensus_methods == "None":
		return printlog("No consensus methods selected, skipping consensus.")

	printlog(f"Applying consensus methods: {consensus_methods}")

	# Load poses data
	if isinstance(poses_input, pd.DataFrame):
		rescored_dataframe = poses_input
	elif isinstance(poses_input, (str, Path)):
		file_path = Path(poses_input)
		if file_path.suffix.lower() == '.csv':
			rescored_dataframe = pd.read_csv(file_path)
		elif file_path.suffix.lower() == '.sdf':
			rescored_dataframe = parallel_SDF_loader(file_path, molColName="Molecule", idName="Pose ID")
		else:
			raise ValueError(f"Unsupported file format: {file_path.suffix}")
	else:
		raise ValueError("Invalid input type for poses_input. Expected DataFrame, string, or Path.")

	# Ensure 'Pose ID' column exists
	if 'Pose ID' not in rescored_dataframe.columns:
		raise ValueError("Input data must contain a 'Pose ID' column")

	# Standardize the scores and add the 'ID' column
	standardized_dataframe = standardize_scores(rescored_dataframe, standardization_type)
	standardized_dataframe["ID"] = standardized_dataframe["Pose ID"].str.split("_").str[0]

	# Rank the scores and add the 'ID' column
	ranked_dataframe = rank_scores(standardized_dataframe)
	ranked_dataframe["ID"] = ranked_dataframe["Pose ID"].str.split("_").str[0]

	# Ensure consensus_methods is a list
	if isinstance(consensus_methods, str):
		consensus_methods = [consensus_methods]

	consensus_results = {}
	for consensus_method in consensus_methods:
		# Check if consensus_method is valid
		if consensus_method not in CONSENSUS_METHODS:
			raise ValueError(f"Invalid consensus method: {consensus_method}")

		# Get the method information from the dictionary
		consensus_info = CONSENSUS_METHODS[consensus_method]
		consensus_type = consensus_info["type"]
		consensus_function = consensus_info["function"]

		# Apply the selected consensus method to the data
		selected_columns = [
			col for col in ranked_dataframe.columns
			if (col not in ["Pose ID", "ID", "Molecule"] and col in list(RESCORING_FUNCTIONS.keys()))]
		if consensus_type == "rank":
			consensus_dataframe = consensus_function(ranked_dataframe, selected_columns)
		elif consensus_type == "score":
			consensus_dataframe = consensus_function(standardized_dataframe, selected_columns)
		else:
			raise ValueError(f"Invalid consensus method type: {consensus_type}")

		# Rename the result column to include the method name
		consensus_dataframe = consensus_dataframe.rename(columns={consensus_dataframe.columns[1]: consensus_method})

		consensus_results[consensus_method] = consensus_dataframe

	# Combine all consensus results
	final_consensus_dataframe = pd.concat(consensus_results.values(), axis=1)
	final_consensus_dataframe = final_consensus_dataframe.loc[:, ~final_consensus_dataframe.columns.duplicated()]

	# Prepare CSV output (with SMILES but without molecules)
	csv_output = final_consensus_dataframe.copy()
	if 'Molecule' in csv_output.columns:
		csv_output['SMILES'] = csv_output['Molecule'].apply(lambda x: Chem.MolToSmiles(x) if x is not None else None)
		csv_output = csv_output.drop(columns=['Molecule'])

	# Prepare SDF output (with molecules)
	sdf_output = final_consensus_dataframe.copy()
	if 'Molecule' not in list(sdf_output.columns) and 'SMILES' in list(sdf_output.columns):
		sdf_output['Molecule'] = sdf_output['SMILES'].apply(Chem.MolFromSmiles)

	# Save output files if output_path is provided
	if output_path:
		output_path = Path(output_path)
		output_path.mkdir(parents=True, exist_ok=True)

		csv_file = output_path / "consensus_results.csv"
		sdf_file = output_path / "consensus_results.sdf"

		csv_output.to_csv(csv_file, index=False)
		PandasTools.WriteSDF(sdf_output,
								str(sdf_file),
								molColName="Molecule",
								idName="ID",
								properties=list(sdf_output.columns))

		printlog(f"Results saved to {csv_file} and {sdf_file}")

	return csv_output, sdf_output


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
			consensus_file = parallel_SDF_loader(w_dir / "consensus" / f"{selection_method}_{consensus_method}_results.sdf",
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
			consensus_file = parallel_SDF_loader(w_dir / "consensus" / f"{selection_method}_{consensus_method}_results.sdf",
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
