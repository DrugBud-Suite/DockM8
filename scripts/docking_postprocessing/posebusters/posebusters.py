import os
import sys
import time
from pathlib import Path

import pandas as pd
from yaml import safe_load
from posebusters import PoseBusters
# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor


def process_chunk(data_chunk, config_path, protein_file):
	# Initialize PoseBusters with the configuration file
	buster = PoseBusters(config=safe_load(open(config_path)))
	data_chunk["mol_cond"] = str(protein_file)
	data_chunk = data_chunk.rename(columns={"Molecule": "mol_pred"})

	# Apply PoseBusters processing and preserve all existing columns
	busted_df = buster.bust_table(data_chunk)
	combined_df = pd.concat([data_chunk.reset_index(drop=True), busted_df.reset_index(drop=True)], axis=1)
	combined_df = combined_df.rename(columns={"mol_pred": "Molecule"})
	cols_to_check = [
		"all_atoms_connected",
		"bond_lengths",
		"bond_angles",
		"internal_steric_clash",
		"aromatic_ring_flatness",
		"double_bond_flatness",
		"protein-ligand_maximum_distance", ]

	# Filter rows based on conditions, but keep all columns
	valid_indices = combined_df[cols_to_check].all(axis=1)
	df_final = combined_df[valid_indices]

	return df_final if df_final.shape[0] > 0 else None


def pose_buster(dataframe: pd.DataFrame, protein_file: Path, n_cpus: int = (os.cpu_count() * 0.9)):
	tic = time.perf_counter()
	printlog("Busting poses...")
	config_path = str(dockm8_path) + "/scripts/docking_postprocessing/posebusters/posebusters_config.yml"

	# Calculate chunk size based on number of CPUs
	chunk_size = len(dataframe) // n_cpus * 8
	if chunk_size == 0:
		chunk_size = 1

	# Split DataFrame into chunks
	chunks = [dataframe[i:i + chunk_size] for i in range(0, len(dataframe), chunk_size)]

	# Use parallel_executor to process chunks
	results = parallel_executor(process_chunk,
								chunks,
								n_cpus,
								job_manager="concurrent_process",
								display_name="PoseBusters",
								config_path=config_path,
								protein_file=protein_file)

	# Filter out None results and concatenate valid DataFrames
	valid_results = [df for df in results if df is not None]
	if valid_results:
		concatenated_df = pd.concat(valid_results, ignore_index=True)
		dataframe = dataframe[dataframe["Pose ID"].isin(concatenated_df["Pose ID"])]
	else:
		dataframe = pd.DataFrame()                     # Return an empty DataFrame if no valid results

	toc = time.perf_counter()
	printlog(f"PoseBusters checking completed in {toc - tic:.2f} seconds.")
	return dataframe
