import os
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import pandas as pd
from rdkit.Chem import PandasTools

# Assuming the same directory structure as in KORP_PL.py
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.file_splitting import split_sdf_single_str
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.utilities import delete_files

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def check_and_download_dligand2():
	dligand2_folder = dockm8_path / "software" / "DLIGAND2"

	if not os.path.exists(dligand2_folder):
		printlog("DLIGAND2 folder not found. Downloading...")
		download_url = "https://github.com/yuedongyang/DLIGAND2/archive/refs/heads/master.zip"
		download_path = dockm8_path / "software" / "DLIGAND2.zip"
		urllib.request.urlretrieve(download_url, download_path)
		printlog("Download complete. Extracting...")

		with zipfile.ZipFile(download_path, 'r') as zip_ref:
			zip_ref.extractall(path=dockm8_path / "software")

		# Rename the extracted folder
		os.rename(dockm8_path / "software" / "DLIGAND2-master", dligand2_folder)

		printlog("Extraction complete. Removing zip file...")
		os.remove(download_path)

		# Change permissions of the executable
		executable_path = dligand2_folder / "bin" / "dligand2.gnu"
		os.chmod(executable_path, os.stat(executable_path).st_mode | stat.S_IEXEC)
		printlog(f"Changed permissions for {executable_path}")

		printlog("DLIGAND2 setup complete.")
	else:
		printlog("DLIGAND2 folder already exists.")

	return dligand2_folder


def DLIGAND2_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs):
	"""
    Rescores a given SDF file using DLIGAND2 software and saves the results to a CSV file.

    Args:
    - sdf (str): The path to the SDF file to be rescored.
    - n_cpus (int): The number of CPUs to use for parallel processing.
    - column_name (str): The name of the column to store the rescored values in.
    - rescoring_folder (str): The path to the folder to store the rescored results.
    - protein_file (str): The path to the protein file.
    - etype (int): The type of potential to use (1 or 2).

    Returns:
    - Path: The path to the CSV file containing the rescoring results.
    """
	rescoring_folder = Path(kwargs.get("rescoring_folder"))
	protein_file = Path(kwargs.get("protein_file"))

	# Check and download DLIGAND2 if necessary
	dligand2_folder = check_and_download_dligand2()

	tic = time.perf_counter()
	(rescoring_folder / f"{column_name}_rescoring").mkdir(parents=True, exist_ok=True)
	split_files_folder = split_sdf_single_str((rescoring_folder / f"{column_name}_rescoring"), sdf)
	split_files_sdfs = [Path(split_files_folder) / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]
	global DLIGAND2_rescoring_splitted

	def DLIGAND2_rescoring_splitted(split_file, protein_file):
		df = PandasTools.LoadSDF(str(split_file), idName="Pose ID", molColName=None)
		df = df[["Pose ID"]]

		with tempfile.NamedTemporaryFile(suffix=".mol2", delete=False) as temp_mol2:
			# Convert SDF to MOL2
			convert_molecules(split_file, Path(temp_mol2.name), "sdf", "mol2")

			dligand2_command = f"cd {dligand2_folder}/bin && ./dligand2.gnu -etype 2 -P {protein_file} -L {temp_mol2.name}"
			process = subprocess.Popen(dligand2_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			stdout, stderr = process.communicate()

			output = stdout.decode().strip()
			try:
				energy = round(float(output), 2)
			except (ValueError, IndexError):
				printlog(f"Warning: Could not parse energy for file {split_file}. Setting to None.")
				energy = None

		# Remove the temporary MOL2 file
		os.unlink(temp_mol2.name)

		df[column_name] = [energy]
		output_csv = str(rescoring_folder / f"{column_name}_rescoring" / (str(split_file.stem) + "_score.csv"))
		df.to_csv(output_csv, index=False)

	# Set the DATAPATH environment variable
	os.environ["DATAPATH"] = str(dligand2_folder / "bin")

	parallel_executor(DLIGAND2_rescoring_splitted, split_files_sdfs, n_cpus, protein_file=protein_file)

	print("Combining DLIGAND2 scores")
	scores_folder = rescoring_folder / f"{column_name}_rescoring"
	score_files = list(scores_folder.glob("*_score.csv"))
	if not score_files:
		print("No CSV files found with names ending in '_score.csv' in the specified folder.")
		return None
	else:
		combined_scores_df = pd.concat([pd.read_csv(file) for file in score_files], ignore_index=True)
		DLIGAND2_rescoring_results = scores_folder / f"{column_name}_scores.csv"
		combined_scores_df.to_csv(DLIGAND2_rescoring_results, index=False)

	delete_files(rescoring_folder / f"{column_name}_rescoring", f"{column_name}_scores.csv")
	toc = time.perf_counter()
	printlog(f"Rescoring with DLIGAND2 complete in {toc-tic:0.4f}!")
	return DLIGAND2_rescoring_results
