import math
import os
import subprocess
import sys
import warnings
from pathlib import Path
import tempfile
import pandas as pd
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.library_preparation.conformer_generation.confgen_RDKit import generate_conformers_RDKit

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def generate_conformers_GypsumDL(df: pd.DataFrame, software: Path, n_cpus: int) -> pd.DataFrame:
	"""
	Generates 3D conformers using GypsumDL.

	Args:
		df (pd.DataFrame): The input DataFrame containing the molecules.
		software (Path): The path to the GypsumDL software.
		n_cpus (int): The number of CPUs to use for the calculation.

	Returns:
		pd.DataFrame: The DataFrame containing the generated conformers.

	Raises:
		Exception: If failed to generate conformers.

	"""
	printlog("Generating 3D conformers using GypsumDL...")

	with tempfile.TemporaryDirectory() as temp_dir:
		temp_dir_path = Path(temp_dir)
		input_sdf = temp_dir_path / "input.sdf"
		output_dir = temp_dir_path / "output"
		output_dir.mkdir(exist_ok=True)

		PandasTools.WriteSDF(df, str(input_sdf), molColName="Molecule", idName="ID")

		# Splitting input SDF file into smaller files for parallel processing
		split_files_folder = split_sdf_str(output_dir / "GypsumDL_split", input_sdf, 10)
		split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

		global gypsum_dl_run

		def gypsum_dl_run(split_file: Path, output_dir: Path, cpus: int):
			results_dir = output_dir / "GypsumDL_results"
			try:
				# Running GypsumDL command for each split file
				gypsum_dl_command = (f"python {software}/gypsum_dl-1.2.1/run_gypsum_dl.py "
										f"-s {split_file} "
										f"-o {results_dir} "
										f"--job_manager multiprocessing "
										f"-p {cpus} "
										f"-m 1 "
										f"-t 10 "
										f"--skip_adding_hydrogen "
										f"--skip_alternate_ring_conformations "
										f"--skip_making_tautomers "
										f"--skip_enumerate_chiral_mol "
										f"--skip_enumerate_double_bonds "
										f"--max_variants_per_compound 1 "
										f"--separate_output_files")
				subprocess.call(gypsum_dl_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			except Exception as e:
				printlog("ERROR: Failed to generate conformers!")
				printlog(e)
			return

		# Running GypsumDL in parallel)
		parallel_executor(gypsum_dl_run,
							list_of_objects=split_files_sdfs,
							n_cpus=3,
							display_name="Conformer Generation with GypsumDL",
							output_dir=output_dir,
							cpus=math.ceil(n_cpus // 3))

		results_dfs = []

		# Loading generated conformers from output directory
		for file in os.listdir(output_dir / "GypsumDL_results"):
			if file.endswith(".sdf"):
				sdf_df = PandasTools.LoadSDF(str(output_dir / "GypsumDL_results" / file),
												molColName="Molecule",
												idName="ID")
				results_dfs.append(sdf_df)

		combined_df = pd.concat(results_dfs)

		# Remove the row containing GypsumDL parameters from the DataFrame
		final_df = combined_df[combined_df["ID"] != "EMPTY MOLECULE DESCRIBING GYPSUM-DL PARAMETERS"]

		# Select only the 'Molecule' and 'ID' columns from the DataFrame
		final_df = final_df[["Molecule", "ID"]]

		# Check if the number of compounds matches the input
		input_compound_count = len(df)
		final_compound_count = len(final_df)

		if final_compound_count != input_compound_count:
			printlog(
				"Conformer generation for some compounds failed. Attempting to generate missing conformers using RDKit..."
			)

			input_ids = set(df["ID"])
			final_ids = set(final_df["ID"])
			missing_ids = input_ids - final_ids
			missing_compounds = df[df["ID"].isin(missing_ids)]

			missing_compounds_conformers = generate_conformers_RDKit(missing_compounds, n_cpus, 'MMFF')

			final_df = pd.concat(final_df, missing_compounds_conformers)

	return final_df
