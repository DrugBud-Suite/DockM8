import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Union

import pandas as pd

# Assuming similar path structure as in the provided code
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog
from scripts.utilities.utilities import delete_files
from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.parallel_executor import parallel_executor


def GenScore_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs):
	"""
    Performs rescoring of ligand poses using the GenScore software package.

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
	pocket_file = str(protein_file).replace(".pdb", "_pocket.pdb")

	(rescoring_folder / f"{column_name}_rescoring").mkdir(parents=True, exist_ok=True)

	if column_name == "GenScore-scoring":
		model = "../trained_models/GatedGCN_ft_1.0_1.pth"
		encoder = "gatedgcn"
	elif column_name == "GenScore-docking":
		model = "../trained_models/GT_0.0_1.pth"
		encoder = "gt"
	elif column_name == "GenScore-balanced":
		model = "../trained_models/GT_ft_0.5_1.pth"
		encoder = "gt"
	else:
		printlog(f"Error: GenScore model not found for column name {column_name}")
		return None

	# Split the input SDF file
	split_files_folder = split_sdf_str(rescoring_folder / f"{column_name}_rescoring", sdf, n_cpus)
	split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]
	global genscore_rescoring_splitted

	def genscore_rescoring_splitted(split_file):
		try:
			# Construct the command for each split file
			cmd = (f"cd {software}/GenScore/example/ &&"
					"conda run -n genscore python genscore.py"
					f" -p {pocket_file}"
					f" -l {split_file}"
					f" -o {rescoring_folder / f'{column_name}_rescoring' / split_file.stem}"
					f" -m {model}"
					f" -e {encoder}")

			# Run GenScore rescoring for each split file
			subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

			return rescoring_folder / f"{column_name}_rescoring" / f"{split_file.stem}.csv"
		except Exception as e:
			printlog(f"Error occurred while running GenScore on {split_file}: {e}")
			return None

	# Run GenScore rescoring in parallel
	rescoring_results = parallel_executor(genscore_rescoring_splitted, split_files_sdfs, n_cpus)

	# Process the results
	genscore_dataframes = []
	for result_file in rescoring_results:
		if result_file and Path(result_file).is_file():
			df = pd.read_csv(result_file)
			genscore_dataframes.append(df)
			# Clean up the split result file
			os.remove(result_file)

	if not genscore_dataframes:
		printlog(f"ERROR: No valid results found for {column_name} rescoring!")
		return None

	genscore_rescoring_results = pd.concat(genscore_dataframes)
	genscore_rescoring_results.rename(columns={"id": "Pose ID", "score": column_name}, inplace=True)
	genscore_scores_path = rescoring_folder / f"{column_name}_rescoring" / f"{column_name}_scores.csv"
	genscore_rescoring_results.to_csv(genscore_scores_path, index=False)

	# Clean up the split files folder
	delete_files(rescoring_folder / f"{column_name}_rescoring", f"{column_name}_scores.csv")

	toc = time.perf_counter()
	printlog(f"Rescoring with {column_name} complete in {toc - tic:0.4f}!")
	return genscore_rescoring_results
