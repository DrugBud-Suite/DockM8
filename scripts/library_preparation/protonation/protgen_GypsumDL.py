import math
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from subprocess import DEVNULL, STDOUT

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.utilities import parallel_executor, printlog, split_sdf_str

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def protonate_GypsumDL(input_sdf: Path, output_dir: Path, software: Path, n_cpus: int):
	"""
    Generates protonation states using GypsumDL.

    Args:
        input_sdf (str): Path to the input SDF file.
        output_dir (str): Path to the output directory.
        software (str): Path to the GypsumDL software.
        n_cpus (int): Number of CPUs to use for the calculation.

    Raises:
        Exception: If failed to generate protomers.

    """
	printlog("Generating protomers using GypsumDL...")
	split_files_folder = split_sdf_str(output_dir / "GypsumDL_split", input_sdf, 10)
	split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

	global gypsum_dl_run

	def gypsum_dl_run(split_file, output_dir, cpus):
		results_dir = output_dir / "GypsumDL_results"
		try:
			gypsum_dl_command = (f"{sys.executable} {software}/gypsum_dl-1.2.1/run_gypsum_dl.py "
									f"-s {split_file} "
									f"-o {results_dir} "
									f"--job_manager multiprocessing "
									f"-p {cpus} "
									f"-m 1 "
									f"-t 10 "
									f"--min_ph 6.5 "
									f"--max_ph 7.5 "
									f"--pka_precision 1 "
									f"--skip_alternate_ring_conformations "
									f"--skip_making_tautomers "
									f"--skip_enumerate_chiral_mol "
									f"--skip_enumerate_double_bonds "
									f"--max_variants_per_compound 1 "
									f"--separate_output_files "
									f"--2d_output_only")
			subprocess.call(gypsum_dl_command, shell=True, stdout=DEVNULL, stderr=STDOUT)
		except Exception as e:
			printlog("ERROR: Failed to generate protomers and conformers!")
			printlog(e)
		return

	parallel_executor(gypsum_dl_run, split_files_sdfs, 3, output_dir=output_dir, cpus=math.ceil(n_cpus // 3))

	results_dfs = []

	for file in os.listdir(output_dir / "GypsumDL_results"):
		if file.endswith(".sdf"):
			sdf_df = PandasTools.LoadSDF(str(output_dir / "GypsumDL_results" / file),
											molColName="Molecule",
											idName="ID")
			results_dfs.append(sdf_df)

	final_df = pd.concat(results_dfs)

	# Remove the row containing GypsumDL parameters from the DataFrame
	final_df = final_df[final_df["ID"] != "EMPTY MOLECULE DESCRIBING GYPSUM-DL PARAMETERS"]

	# Select only the 'Molecule' and 'ID' columns from the DataFrame
	final_df = final_df[["Molecule", "ID"]]

	# Load the input SDF file and count the number of compounds
	input_df = PandasTools.LoadSDF(str(input_sdf), molColName="Molecule", idName="ID")
	input_compound_count = len(input_df)

	# Check if the number of compounds in final_df matches the input
	final_compound_count = len(final_df)

	if final_compound_count != input_compound_count:
		printlog(
			"Some compounds were not able to be protonated. Adding those compounds using the input SDF file instead.")
		input_ids = set(input_df["ID"])
		final_ids = set(final_df["ID"])
		missing_ids = input_ids - final_ids
		missing_compounds = input_df[input_df["ID"].isin(missing_ids)]
		final_df = pd.concat([final_df, missing_compounds])

	output_file = output_dir / "protonated_library.sdf"

	PandasTools.WriteSDF(final_df, str(output_file), molColName="Molecule", idName="ID")
	shutil.rmtree(output_dir / "GypsumDL_results", ignore_errors=True)
	shutil.rmtree(output_dir / "GypsumDL_split", ignore_errors=True)

	(output_dir / "gypsum_dl_success.sdf").unlink(missing_ok=True)
	(output_dir / "gypsum_dl_failed.smi").unlink(missing_ok=True)

	return output_file
