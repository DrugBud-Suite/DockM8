import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from subprocess import DEVNULL, STDOUT

import pandas as pd
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


def gnina_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs):
	"""
    Performs rescoring of ligand poses using the gnina software package. The function splits the input SDF file into
    smaller files, and then runs gnina on each of these files in parallel. The results are then combined into a single
    Pandas dataframe and saved to a CSV file.

    Args:
        sdf (str): The path to the input SDF file.
        n_cpus (int): The number of CPUs to use for parallel execution.
        column_name (str): The name of the column in the output dataframe that will contain the rescoring results.

    Returns:
        A Pandas dataframe containing the rescoring results.
    """
	tic = time.perf_counter()
	rescoring_folder = kwargs.get("rescoring_folder")
	software = kwargs.get("software")
	protein_file = kwargs.get("protein_file")
	cnn = "crossdock_default2018"
	split_files_folder = split_sdf_str(rescoring_folder / f"{column_name}_rescoring", sdf, n_cpus)
	split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

	global gnina_rescoring_splitted

	def gnina_rescoring_splitted(split_file, protein_file):
		gnina_folder = rescoring_folder / f"{column_name}_rescoring"
		results = gnina_folder / f"{Path(split_file).stem}_{column_name}.sdf"
		gnina_cmd = (f"{software}/gnina"
			f" --receptor {protein_file}"
			f" --ligand {split_file}"
			f" --out {results}"
			" --cpu 1"
			" --score_only"
			f" --cnn {cnn} --no_gpu")
		try:
			subprocess.call(gnina_cmd, shell=True, stdout=DEVNULL, stderr=STDOUT)
		except Exception as e:
			printlog(f"{column_name} rescoring failed: " + e)
		return

	parallel_executor(gnina_rescoring_splitted, split_files_sdfs, n_cpus, protein_file=protein_file)

	try:
		gnina_dataframes = [
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
		gnina_rescoring_results = pd.concat(gnina_dataframes)
	except Exception as e:
		printlog(f"ERROR: Could not combine {column_name} rescored poses")
		printlog(e)
	gnina_rescoring_results.rename(columns={
		"minimizedAffinity": "GNINA-Affinity", "CNNscore": "CNN-Score", "CNNaffinity": "CNN-Affinity"},
			inplace=True)

	gnina_rescoring_results = gnina_rescoring_results[["Pose ID", column_name]]
	gnina_scores_path = rescoring_folder / f"{column_name}_rescoring" / f"{column_name}_scores.csv"
	gnina_rescoring_results.to_csv(gnina_scores_path, index=False)
	delete_files(rescoring_folder / f"{column_name}_rescoring", f"{column_name}_scores.csv")
	toc = time.perf_counter()
	printlog(f"Rescoring with {column_name} complete in {toc - tic:0.4f}!")
	return gnina_rescoring_results
