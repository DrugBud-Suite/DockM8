import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, PandasTools

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

# yapf: disable
RESCORING_FUNCTIONS = {
	"AAScore": {"class": AAScore, "column_name": "AAScore", "best_value": "min", "score_range": (100, -100)},
	"AD4": {"class": AD4, "column_name": "AD4", "best_value": "min", "score_range": (1000, -100)},
	"CENsible": {"class": CENsible, "column_name": "CENsible", "best_value": "max", "score_range": (0, 20)},
	"CHEMPLP": {"class": CHEMPLP, "column_name": "CHEMPLP", "best_value": "min", "score_range": (200, -200)},
	"ConvexPLR": {"class": ConvexPLR, "column_name": "ConvexPLR", "best_value": "max", "score_range": (-10, 10)},
	"DLIGAND2": {"class": DLIGAND2, "column_name": "DLIGAND2", "best_value": "min", "score_range": (100, -200)},
	"ITScoreAff": {"class": ITScoreAff, "column_name": "ITScoreAff", "best_value": "min", "score_range": (100, -200)},
	"GNINA-Affinity": {"class": partial(Gnina, score_type="affinity"), "column_name": "GNINA-Affinity", "best_value": "min", "score_range": (100, -100)},
	"CNN-Score": {"class": partial(Gnina, score_type="cnn_score"), "column_name": "CNN-Score", "best_value": "max", "score_range": (0, 1)},
	"CNN-Affinity": {"class": partial(Gnina, score_type="cnn_affinity"), "column_name": "CNN-Affinity", "best_value": "max", "score_range": (0, 20)},
	"GenScore-scoring": {"class": partial(GenScore, score_type="scoring"), "column_name": "GenScore-scoring", "best_value": "max", "score_range": (0, 100)},
	"GenScore-docking": {"class": partial(GenScore, score_type="docking"), "column_name": "GenScore-docking", "best_value": "max", "score_range": (0, 100)},
	"GenScore-balanced": {"class": partial(GenScore, score_type="balanced"), "column_name": "GenScore-balanced", "best_value": "max", "score_range": (0, 100)},
	"KORP-PL": {"class": KORPL, "column_name": "KORP-PL", "best_value": "min", "score_range": (200, -500)},
	"LinF9": {"class": LinF9, "column_name": "LinF9", "best_value": "min", "score_range": (50, -50)},
	"NNScore": {"class": NNScore, "column_name": "NNScore", "best_value": "max", "score_range": (0, 20)},
	"PANTHER": {"class": partial(PANTHER, score_type="PANTHER"), "column_name": "PANTHER", "best_value": "max", "score_range": (0, 1)},
	"PANTHER-ESP": {"class": partial(PANTHER, score_type="PANTHER-ESP"), "column_name": "PANTHER-ESP", "best_value": "max", "score_range": (0, 1)},
	"PANTHER-Shape": {"class": partial(PANTHER, score_type="PANTHER-Shape"), "column_name": "PANTHER-Shape", "best_value": "max", "score_range": (0, 1)},
	"PLECScore": {"class": PLECScore, "column_name": "PLECScore", "best_value": "max", "score_range": (0, 20)},
	"PLP": {"class": PLP, "column_name": "PLP", "best_value": "min", "score_range": (200, -200)},
	"RFScoreVS": {"class": RFScoreVS, "column_name": "RFScoreVS", "best_value": "max", "score_range": (5, 10)},
	"RTMScore": {"class": RTMScore, "column_name": "RTMScore", "best_value": "max", "score_range": (0, 100)},
	"SCORCH": {"class": SCORCH, "column_name": "SCORCH", "best_value": "max", "score_range": (0, 1)},
	"Vinardo": {"class": Vinardo, "column_name": "Vinardo", "best_value": "min", "score_range": (200, -20)}
}
# yapf: enable


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
    RDLogger.DisableLog("rdApp.*")
    tic = time.perf_counter()

    # Process input
    temp_dir = create_temp_dir("rescore_poses")
    try:
        if isinstance(poses, Path):
            sdf = poses
            original_poses = parallel_SDF_loader(sdf, molColName='Molecule', idName='Pose ID')
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

        # Sort original_poses by 'Pose ID' to ensure consistent ordering
        original_poses = original_poses.sort_values('Pose ID')

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
            combined_df = pd.merge(original_poses, existing_results, on="Pose ID", how="left").sort_values('Pose ID')
        elif output_file:
            original_poses['Pose ID'].to_csv(output_file, index=False)
            combined_df = original_poses

        for function in functions_to_run:
            scoring_function_class = RESCORING_FUNCTIONS.get(function)['class']
            if scoring_function_class:
                try:
                    scoring_function: ScoringFunction = scoring_function_class(software_path=software)
                    result = scoring_function.rescore(sdf,
                                                      n_cpus,
                                                      protein_file=protein_file,
                                                      pocket_definition=pocket_definition)
                    combined_df = pd.merge(combined_df, result, on="Pose ID", how="left").sort_values('Pose ID')
                    if output_file:
                        columns_to_write = [col for col in combined_df.columns if col != 'Molecule']
                        combined_df[columns_to_write].to_csv(output_file, index=False)
                except Exception as e:
                    printlog(f"Failed for {function} : {e}")
                    printlog(traceback.format_exc())
            else:
                printlog(f"Unknown scoring function: {function}")

        # Ensure required columns are present
        if 'ID' not in combined_df.columns:
            combined_df['ID'] = combined_df['Pose ID'].str.split("_").str[0]
        if 'SMILES' not in combined_df.columns:
            combined_df['SMILES'] = combined_df['Molecule'].apply(lambda x: Chem.MolToSmiles(x)
                                                                  if x is not None else None)

        # Reorder columns for CSV output
        csv_columns = ['Pose ID', 'ID', 'SMILES'] + [
            col for col in combined_df.columns if col not in ['Pose ID', 'ID', 'SMILES', 'Molecule']]
        csv_df = combined_df[csv_columns].sort_values('Pose ID')

        # Convert columns to float where possible
        for col in csv_df.columns:
            if col not in ["Pose ID", "ID", "SMILES"]:
                csv_df[col] = pd.to_numeric(csv_df[col], errors='coerce')

        # Write final CSV
        if output_file:
            csv_df.to_csv(output_file, index=False)
            if len(functions_to_run) > 0:
                printlog(f"Final rescored poses CSV written to: {output_file}")

        # Reorder columns for SDF output
        sdf_columns = ['Pose ID', 'ID', 'SMILES', 'Molecule'] + [
            col for col in combined_df.columns if col not in ['Pose ID', 'ID', 'SMILES', 'Molecule']]
        sdf_df = combined_df[sdf_columns].sort_values('Pose ID')

        # Write final SDF
        if output_file:
            PandasTools.WriteSDF(sdf_df,
                                 str(output_file.with_suffix(".sdf")),
                                 molColName='Molecule',
                                 idName='Pose ID',
                                 properties=list(sdf_df.columns))
            if len(functions_to_run) > 0:
                printlog(f"Final rescored poses SDF written to: {output_file.with_suffix('.sdf')}")
            return output_file
        toc = time.perf_counter()
        printlog(f"Rescoring complete in {toc - tic:0.4f}!")

        return combined_df.sort_values('Pose ID')

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
		scoring_function_class = RESCORING_FUNCTIONS.get(function)['class']

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
