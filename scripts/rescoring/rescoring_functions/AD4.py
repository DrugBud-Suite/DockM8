import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from subprocess import DEVNULL, STDOUT

import pandas as pd
from pandas import DataFrame
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog
from scripts.utilities.utilities import delete_files
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.file_splitting import split_sdf_str

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def AD4_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs) -> DataFrame:
	"""
    Performs rescoring of poses using the AutoDock4 (AD4) scoring function.

    Args:
        sdf (str): The path to the input SDF file containing the poses to be rescored.
        n_cpus (int): The number of CPUs to be used for the rescoring process.
        column_name (str): The name of the column in the output dataframe to store the AD4 scores.
        kwargs: Additional keyword arguments including rescoring_folder, software, protein_file, and pocket_de.

    Returns:
        DataFrame: A dataframe containing the 'Pose ID' and AD4 score columns for the rescored poses.
    """
	tic = time.perf_counter()

	rescoring_folder = kwargs.get("rescoring_folder")
	software = kwargs.get("software")
	protein_file = kwargs.get("protein_file")

	split_files_folder = split_sdf_str(rescoring_folder / f"{column_name}_rescoring", sdf, n_cpus)
	split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

	AD4_rescoring_folder = rescoring_folder / f"{column_name}_rescoring"
	AD4_rescoring_folder.mkdir(parents=True, exist_ok=True)

	global AD4_rescoring_splitted

	def AD4_rescoring_splitted(split_file, protein_file):
		AD4_rescoring_folder = rescoring_folder / f"{column_name}_rescoring"
		results = AD4_rescoring_folder / f"{Path(split_file).stem}_{column_name}.sdf"
		AD4_cmd = (f"{software}/gnina"
			f" --receptor {protein_file}"
			f" --ligand {split_file}"
			f" --out {results}"
			" --score_only"
			" --scoring ad4_scoring"
			" --cnn_scoring none")
		try:
			subprocess.call(AD4_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
		except Exception as e:
			printlog(f"{column_name} rescoring failed: " + e)
		return

	parallel_executor(AD4_rescoring_splitted, split_files_sdfs, n_cpus, protein_file=protein_file)

	try:
		AD4_dataframes = [
			PandasTools.LoadSDF(str(rescoring_folder / f"{column_name}_rescoring" / file),
				idName="Pose ID",
				molColName=None,
				includeFingerprints=False,
				embedProps=False)
			for file in os.listdir(rescoring_folder / f"{column_name}_rescoring")
			if file.startswith("split") and file.endswith(".sdf")]
	except Exception as e:
		printlog(f"ERROR: Failed to Load {column_name} rescoring SDF file!")
		printlog(e)
	try:
		AD4_rescoring_results = pd.concat(AD4_dataframes)
	except Exception as e:
		printlog(f"ERROR: Could not combine {column_name} rescored poses")
		printlog(e)

	AD4_rescoring_results.rename(columns={"minimizedAffinity": column_name}, inplace=True)
	AD4_rescoring_results = AD4_rescoring_results[["Pose ID", column_name]]
	AD4_scores_file = AD4_rescoring_folder / f"{column_name}_scores.csv"
	AD4_rescoring_results.to_csv(AD4_scores_file, index=False)
	delete_files(AD4_rescoring_folder, f"{column_name}_scores.csv")
	toc = time.perf_counter()
	printlog(f"Rescoring with AD4 complete in {toc-tic:0.4f}!")
	return AD4_rescoring_results
