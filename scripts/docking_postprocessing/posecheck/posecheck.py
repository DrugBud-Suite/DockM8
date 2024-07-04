import os
import sys
import time
from pathlib import Path

import pandas as pd
from posecheck import PoseCheck

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

import multiprocessing

from scripts.utilities.logging import printlog


def pose_checker(dataframe: pd.DataFrame,
		protein_file: Path,
		clash_cutoff: int = 5,
		strain_cutoff: int = 5000,
		n_cpus: int = int(os.cpu_count() * 0.9),
	):
	try:
		tic = time.perf_counter()
		printlog("Checking poses using PoseCheck...")
		pc = PoseCheck()
		pc.load_protein_from_pdb(str(protein_file))
		# Adjust the process_row function to work with dictionaries
		global process_row

		def process_row(row):
			mol = [row["Molecule"]]
			pc.load_ligands_from_mols(mol)
			clashes = pc.calculate_clashes()
			strain = pc.calculate_strain_energy()
			row["clashes"] = clashes[0]
			row["strain"] = round(strain[0], 3)
			return row

		# Convert DataFrame to a list of dicts, which is compatible with multiprocessing
		rows = dataframe.to_dict("records")

		# Create a pool of worker processes
		pool = multiprocessing.Pool(n_cpus)

		# Use the pool to process rows in parallel
		processed_rows = pool.map(process_row, rows)

		# Close the pool and wait for the work to finish
		pool.close()
		pool.join()

		# Convert list of dicts back to DataFrame
		dataframe = pd.DataFrame(processed_rows)

		# Filter the dataframe based on clash and strain cutoffs
		dataframe = dataframe[dataframe["clashes"] <= clash_cutoff]
		dataframe = dataframe[dataframe["strain"] <= strain_cutoff]

		toc = time.perf_counter()
		printlog(f"Pose checking completed in {toc - tic:.2f} seconds")
	except Exception as e:
		printlog("ERROR: Failed to check poses with PoseCheck!")
		printlog(str(e))

	return dataframe.drop(columns=["clashes", "strain"])
