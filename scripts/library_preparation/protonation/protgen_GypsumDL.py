import math
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.utilities import parallel_SDF_loader
from scripts.setup.software_manager import ensure_software_installed

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def protonate_GypsumDL(df: pd.DataFrame,
	software: Path,
	n_cpus: int,
	min_ph: float = 6.5,
	max_ph: float = 7.5,
	pka_precision: float = 1.0) -> pd.DataFrame:
	"""
	Generates protonation states using GypsumDL.

	Args:
		df (pd.DataFrame): Input DataFrame containing molecules.
		software (Path): Path to the GypsumDL software.
		n_cpus (int): Number of CPUs to use for the calculation.
		min_ph (float, optional): Minimum pH for protonation. Defaults to 6.5.
		max_ph (float, optional): Maximum pH for protonation. Defaults to 7.5.
		pka_precision (float, optional): pKa precision for protonation. Defaults to 1.0.

	Returns:
		pd.DataFrame: DataFrame with protonated molecules.

	Raises:
		Exception: If failed to generate protomers.
	"""
	printlog("Generating protomers using GypsumDL...")
	ensure_software_installed("GYPSUM_DL", software)
	with tempfile.TemporaryDirectory() as temp_dir:
		temp_dir_path = Path(temp_dir)
		input_sdf = temp_dir_path / "input.sdf"
		output_dir = temp_dir_path / "output"
		output_dir.mkdir(exist_ok=True)

		# Write input DataFrame to SDF
		PandasTools.WriteSDF(df, str(input_sdf), molColName="Molecule", idName="ID")

		# Split SDF function (you'll need to implement this)
		split_files_folder = split_sdf_str(output_dir / "GypsumDL_split", input_sdf, 10)
		split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

		global gypsum_dl_run

		def gypsum_dl_run(split_file: Path, output_dir: Path, cpus: int):
			results_dir = output_dir / "GypsumDL_results"
			try:
				gypsum_dl_command = (f"{sys.executable} {software}/gypsum_dl-1.2.1/run_gypsum_dl.py "
					f"-s {split_file} "
					f"-o {results_dir} "
					f"--job_manager multiprocessing "
					f"-p {cpus} "
					f"-m 1 "
					f"-t 10 "
					f"--min_ph {min_ph} "
					f"--max_ph {max_ph} "
					f"--pka_precision {pka_precision} "
					f"--skip_alternate_ring_conformations "
					f"--skip_making_tautomers "
					f"--skip_enumerate_chiral_mol "
					f"--skip_enumerate_double_bonds "
					f"--max_variants_per_compound 1 "
					f"--separate_output_files "
					f"--2d_output_only")
				subprocess.call(gypsum_dl_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			except Exception as e:
				printlog(f"ERROR: Failed to generate protomers! {e}")
			return

		# Parallel execution function (you'll need to implement this)
		parallel_executor(gypsum_dl_run,
				list_of_objects=split_files_sdfs,
				n_cpus=3,
				display_name="Protomer Generation with GypsumDL",
				output_dir=output_dir,
				cpus=math.ceil(n_cpus // 3))

		results_dfs = []

		for file in os.listdir(output_dir / "GypsumDL_results"):
			if file.endswith(".sdf"):
				sdf_df = parallel_SDF_loader(output_dir / "GypsumDL_results" / file, molColName="Molecule", idName="ID")
				results_dfs.append(sdf_df)

		final_df = pd.concat(results_dfs)

		# Remove the row containing GypsumDL parameters from the DataFrame
		final_df = final_df[final_df["ID"] != "EMPTY MOLECULE DESCRIBING GYPSUM-DL PARAMETERS"]

		# Select only the 'Molecule' and 'ID' columns from the DataFrame
		final_df = final_df[["Molecule", "ID"]]

		# Check if the number of compounds in final_df matches the input
		input_compound_count = len(df)
		final_compound_count = len(final_df)

		if final_compound_count != input_compound_count:
			printlog(
				"Some compounds were not able to be protonated. Adding those compounds using the input DataFrame instead."
			)
			input_ids = set(df["ID"])
			final_ids = set(final_df["ID"])
			missing_ids = input_ids - final_ids
			missing_compounds = df[df["ID"].isin(missing_ids)]
			final_df = pd.concat([final_df, missing_compounds])

	return final_df
