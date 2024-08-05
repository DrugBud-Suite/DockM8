import os
import shutil
import sys
import tempfile
import time
import warnings
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import AllChem, PandasTools
import traceback

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring_functions.AAScore import AAScore
from scripts.rescoring.rescoring_functions.AD4 import AD4
from scripts.rescoring.rescoring_functions.CENsible import CENsible
from scripts.rescoring.rescoring_functions.CHEMPLP import CHEMPLP
from scripts.rescoring.rescoring_functions.ConvexPLR import ConvexPLR
from scripts.rescoring.rescoring_functions.DLIGAND2 import DLIGAND2
from scripts.rescoring.rescoring_functions.GenScore import GenScore
from scripts.rescoring.rescoring_functions.gnina import Gnina
from scripts.rescoring.rescoring_functions.ITScoreAff import ITScoreAff
from scripts.rescoring.rescoring_functions.KORP_PL import KORPL
from scripts.rescoring.rescoring_functions.LinF9 import LinF9
from scripts.rescoring.rescoring_functions.NNScore import NNScore
from scripts.rescoring.rescoring_functions.PANTHER import PANTHER
from scripts.rescoring.rescoring_functions.PLECScore import PLECScore
from scripts.rescoring.rescoring_functions.PLP import PLP
from scripts.rescoring.rescoring_functions.RFScoreVS import RFScoreVS
from scripts.rescoring.rescoring_functions.RTMScore import RTMScore
from scripts.rescoring.rescoring_functions.SCORCH import SCORCH
from scripts.rescoring.rescoring_functions.vinardo import Vinardo
from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog
from scripts.utilities.utilities import parallel_SDF_loader

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Updated RESCORING_FUNCTIONS dictionary using class instances

RESCORING_FUNCTIONS = {
	"AAScore": AAScore,
	"AD4": AD4,
	"CENsible": CENsible,
	"CHEMPLP": CHEMPLP,
	"CNN-Affinity": partial(Gnina, score_type="cnn_affinity"),
	"CNN-Score": partial(Gnina, score_type="cnn_score"),
	"ConvexPLR": ConvexPLR,
	"DLIGAND2": DLIGAND2,
	"GenScore-balanced": partial(GenScore, score_type="balanced"),
	"GenScore-docking": partial(GenScore, score_type="docking"),
	"GenScore-scoring": partial(GenScore, score_type="scoring"),
	"GNINA-Affinity": partial(Gnina, score_type="affinity"),
	"ITScoreAff": ITScoreAff,
	"KORP-PL": KORPL,
	"LinF9": LinF9,
	"NNScore": NNScore,
	"PANTHER": partial(PANTHER, score_type="PANTHER"),
	"PANTHER-ESP": partial(PANTHER, score_type="PANTHER-ESP"),
	"PANTHER-Shape": partial(PANTHER, score_type="PANTHER-Shape"),
	"PLECScore": PLECScore,
	"PLP": PLP,
	"RFScoreVS": RFScoreVS,
	"RTMScore": RTMScore,
	"SCORCH": SCORCH,
	"Vinardo": Vinardo, }


def create_temp_dir(name: str) -> Path:
	"""
	Creates a temporary directory for the scoring function.

	Args:
		name (str): The name of the scoring function.

	Returns:
		Path: The path to the temporary directory.
	"""
	os.makedirs(Path.home() / "dockm8_temp_files", exist_ok=True)
	return Path(tempfile.mkdtemp(dir=Path.home() / "dockm8_temp_files", prefix=f"dockm8_{name}_"))


def remove_temp_dir(temp_dir: Path):
	"""
	Removes the temporary directory.

	Args:
		temp_dir (Path): The path to the temporary directory.
	"""
	shutil.rmtree(str(temp_dir), ignore_errors=True)


def rescore_poses(protein_file: Path,
					pocket_definition: dict,
					software: Path,
					poses: Union[Path, pd.DataFrame],
					functions: List[str],
					n_cpus: int,
					output_file: Optional[Path] = None) -> pd.DataFrame:
	"""
    Rescores poses using specified scoring functions. If an output file is provided and exists,
    it will only run the missing scoring functions. Saves results to CSV and SDF incrementally
    if output_file is specified, and adds a SMILES column to the output.

    Args:
        protein_file (Path): Path to the protein file.
        pocket_definition (dict): Dictionary defining the pocket.
        software (Path): Path to the software.
        poses (Union[Path, pd.DataFrame]): Path to the clustered SDF file or DataFrame of poses.
        functions (List[str]): List of scoring function names.
        n_cpus (int): Number of CPUs to use for parallel processing.
        output_file (Optional[Path], optional): Path to the output CSV file. Defaults to None.

    Returns:
        pd.DataFrame: Combined DataFrame of the rescored poses, including SMILES.
    """
	RDLogger.DisableLog("rdApp.*")
	tic = time.perf_counter()

	# Process input
	temp_dir = create_temp_dir("rescore_poses")
	try:
		if isinstance(poses, Path):
			sdf = poses
			original_poses = parallel_SDF_loader(sdf, molColName='Molecule', idName='Pose ID', removeHs=False)
		elif isinstance(poses, pd.DataFrame):
			sdf = temp_dir / "temp_clustered.sdf"
			original_poses = poses.copy()
			PandasTools.WriteSDF(original_poses,
									sdf,
									molColName='Molecule',
									idName='Pose ID',
									properties=list(original_poses.columns))
		else:
			raise ValueError("poses must be a Path or DataFrame")

		# Add SMILES column to original_poses
		original_poses['SMILES'] = original_poses['Molecule'].apply(lambda x: AllChem.MolToSmiles(x)
																	if x is not None else None)

		existing_results = pd.DataFrame()
		functions_to_run = functions.copy()

		if output_file and output_file.exists():
			existing_results = pd.read_csv(output_file)
			existing_results = existing_results[[
				col for col in existing_results.columns if col in list(RESCORING_FUNCTIONS.keys()) + ["Pose ID"]]]
			printlog(f"Found existing results in {output_file}")
			existing_functions = [
				col for col in existing_results.columns if col != "Pose ID" and col in list(RESCORING_FUNCTIONS.keys())]
			functions_to_run = [f for f in functions if f not in existing_functions]
			printlog(f"Functions to run: {', '.join(functions_to_run)}")
		elif output_file:
			# Initialize output file with original poses if output_file is specified but doesn't exist
			columns_to_write = ['Pose ID'] + [
				col for col in original_poses.columns if col not in ['Pose ID', 'SMILES', 'Molecule']]
			original_poses[columns_to_write].to_csv(output_file, index=False)

		combined_df = pd.merge(original_poses, existing_results, on="Pose ID", how="left")

		skipped_functions = []
		for function in functions_to_run:
			scoring_function_class = RESCORING_FUNCTIONS.get(function)
			if scoring_function_class:
				try:
					scoring_function: ScoringFunction = scoring_function_class(software_path=software)
					result = scoring_function.rescore(sdf,
														n_cpus,
														protein_file=protein_file,
														pocket_definition=pocket_definition)
					print(result)
					# Merge new resul	ts with combined_df
					combined_df = pd.merge(combined_df, result, on="Pose ID", how="left")

					# Write updated results to CSV if output_file is specified
					if output_file:
						columns_to_write = [col for col in combined_df.columns if col != 'Molecule']
						combined_df[columns_to_write].to_csv(output_file, index=False)
						printlog(f"Updated results with {function} written to {output_file}")
				except Exception as e:
					printlog(f"Failed for {function} : {e}")
					printlog(traceback.format_exc())
			else:
				skipped_functions.append(function)

		if skipped_functions:
			printlog(f'Skipping functions: {", ".join(skipped_functions)}')

		# Ensure "Pose ID" is the first column, followed by "SMILES"
		columns = combined_df.columns.tolist()
		columns.insert(0, columns.pop(columns.index("Pose ID")))
		combined_df = combined_df[columns]

		# Convert columns to float where possible
		for col in combined_df.columns:
			if col not in ["Pose ID", "Molecule", "SMILES"]:
				combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

		# Write final SDF if output_file is specified
		if output_file:
			PandasTools.WriteSDF(combined_df,
					str(output_file.with_suffix(".sdf")),
					molColName='Molecule',
					idName='Pose ID',
					properties=list(combined_df.columns))
			printlog(f"Final rescored poses written to : {output_file} and {output_file.with_suffix('.sdf')}")

		toc = time.perf_counter()
		printlog(f"Rescoring complete in {toc - tic:0.4f}!")

		return combined_df

	finally:
		remove_temp_dir(temp_dir)


def rescore_docking(poses: Union[Path, pd.DataFrame],
		protein_file: Path,
		pocket_definition: dict,
		software: Path,
		function: str,
		n_cpus: int) -> pd.DataFrame:
	"""
	Rescores the docking poses using a specified scoring function.

	Args:
		poses (Union[Path, pd.DataFrame]): The path to the SDF file containing the docking poses or a DataFrame of poses.
		protein_file (Path): The path to the protein file.
		pocket_definition (dict): A dictionary defining the pocket for docking.
		software (Path): The path to the software used for docking.
		function (str): The name of the scoring function to use for rescoring.
		n_cpus (int): The number of CPUs to use for parallel processing.

	Returns:
		pd.DataFrame: A DataFrame containing the best poses after rescoring.

	Raises:
		ValueError: If an unknown scoring function is specified.
	"""
	RDLogger.DisableLog("rdApp.*")
	tic = time.perf_counter()

	temp_dir = create_temp_dir("rescore_docking")
	try:
		# Process input
		if isinstance(poses, Path):
			sdf = poses
		elif isinstance(poses, pd.DataFrame):
			sdf = temp_dir / "temp_clustered.sdf"
			PandasTools.WriteSDF(poses, sdf, molColName='Molecule', idName='Pose ID', properties=list(poses.columns))
		else:
			raise ValueError("poses must be a Path or DataFrame")
		scoring_function_class = RESCORING_FUNCTIONS.get(function)

		if scoring_function_class is None:
			raise ValueError(f"Unknown scoring function: {function}")

		scoring_function = scoring_function_class(software_path=software)
		score_df = scoring_function.rescore(str(sdf),
					n_cpus,
					protein_file=protein_file,
					pocket_definition=pocket_definition)

		score_df["Pose_Number"] = score_df["Pose ID"].str.split("_").str[2].astype(int)
		score_df["Docking_program"] = score_df["Pose ID"].str.split("_").str[1].astype(str)
		score_df["ID"] = score_df["Pose ID"].str.split("_").str[0].astype(str)

		if scoring_function.best_value == "min":
			best_pose_indices = score_df.groupby("ID")[scoring_function.column_name].idxmin()
		else:
			best_pose_indices = score_df.groupby("ID")[scoring_function.column_name].idxmax()

		best_poses = pd.DataFrame(score_df.loc[best_pose_indices, "Pose ID"])

		toc = time.perf_counter()
		printlog(f"Rescoring complete in {toc - tic:0.4f}!")
		return best_poses

	finally:
		remove_temp_dir(temp_dir)
