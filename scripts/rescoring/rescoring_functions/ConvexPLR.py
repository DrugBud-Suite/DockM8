import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

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


def ConvexPLR_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs):
	"""
    Rescores the given SDF file using Convex-PLR software and saves the results in a CSV file.

    Args:
        - sdf (str): path to the input SDF file
        - n_cpus (int): number of CPUs to use for parallel processing
        - column_name (str): name of the column to store the scores in the output CSV file
        - rescoring_folder (str): path to the folder to store the rescoring results
        - software (str): path to the Convex-PLR software
        - protein_file (str): path to the protein file
        - pocket_definition (str): path to the pocket definition file

    Returns:
        - None
    """

	tic = time.perf_counter()
	rescoring_folder = kwargs.get("rescoring_folder")
	software = kwargs.get("software")
	protein_file = kwargs.get("protein_file")

	(rescoring_folder / f"{column_name}_rescoring").mkdir(parents=True, exist_ok=True)
	split_files_folder = split_sdf_str((rescoring_folder / f"{column_name}_rescoring"), sdf, n_cpus)
	split_files_sdfs = [Path(split_files_folder) / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]
	global ConvexPLR_rescoring_splitted

	def ConvexPLR_rescoring_splitted(split_file, protein_file):
		df = PandasTools.LoadSDF(str(split_file), idName="Pose ID", molColName=None)
		df = df[["Pose ID"]]
		ConvexPLR_command = (f"{software}/Convex-PL" + f" --receptor {protein_file}" + f" --ligand {split_file}" +
				" --sdf --regscore")
		process = subprocess.Popen(ConvexPLR_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
		stdout, stderr = process.communicate()
		energies = []
		output = stdout.decode().splitlines()
		for line in output:
			if line.startswith("model"):
				parts = line.split(",")
				energy = round(float(parts[1].split("=")[1]), 2)
				energies.append(energy)
		df[column_name] = energies
		output_csv = str(rescoring_folder / f"{column_name}_rescoring" / (str(split_file.stem) + "_scores.csv"))
		df.to_csv(output_csv, index=False)
		return

	parallel_executor(ConvexPLR_rescoring_splitted, split_files_sdfs, n_cpus, protein_file=protein_file)

	# Get a list of all files with names ending in "_scores.csv"
	score_files = list((rescoring_folder / f"{column_name}_rescoring").glob("*_scores.csv"))
	# Read and concatenate the CSV files into a single DataFrame
	combined_scores_df = pd.concat([pd.read_csv(file) for file in score_files], ignore_index=True)
	# Save the combined scores to a single CSV file
	ConvexPLR_rescoring_results = rescoring_folder / f"{column_name}_rescoring" / f"{column_name}_scores.csv"
	combined_scores_df.to_csv(ConvexPLR_rescoring_results, index=False)
	delete_files(rescoring_folder / f"{column_name}_rescoring", f"{column_name}_scores.csv")
	toc = time.perf_counter()
	printlog(f"Rescoring with ConvexPLR complete in {toc-tic:0.4f}!")
	return ConvexPLR_rescoring_results
