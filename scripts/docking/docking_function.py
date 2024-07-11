import math
import os
import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor


class DockingFunction(ABC):

	"""
	Abstract base class for docking functions used in DockM8.
	"""

	def __init__(self, name: str, software_path: Path):
		self.name = name
		self.software_path = software_path

	@abstractmethod
	def process_docking_result(self, result_file: Path, n_poses: int) -> pd.DataFrame:
		"""
		Processes a single docking result file and returns a DataFrame.
		"""
		pass

	def dock(self,
		library: Union[pd.DataFrame, Path],
		protein_file: Path,
		pocket_definition: dict,
		exhaustiveness: int,
		n_poses: int,
		n_cpus: int,
		job_manager: str = "concurrent_process",
		output_sdf: Path = None) -> pd.DataFrame:
		"""
		Performs docking using the specific docking software and returns results as a DataFrame.
		"""
		temp_dir = self.create_temp_dir()
		try:
			# Handle input as DataFrame or file
			if isinstance(library, pd.DataFrame):
				library_path = temp_dir / "temp_library.sdf"
				PandasTools.WriteSDF(library, str(library_path), molColName="Molecule", idName="ID")
			else:
				library_path = library

			if self.name in ["FABIND+", "PLANTAIN"]:
				if n_cpus == 1:
					pass
				else:
					n_cpus = 2

			# Generate batches of ligands
			if self.name in ["FABIND+", "PLANTAIN"]:
				batches = self._create_batches(library_path, n_cpus)
			else:
				batches = self._create_batches(library_path, n_cpus * 8)
			# Adjust n_cpus if there are fewer batches than CPUs
			n_cpus_to_use = min(n_cpus, len(batches))

			printlog(f"Docking with {self.name}.")
			# Perform docking
			results = parallel_executor(self.dock_batch,
										batches,
										n_cpus=n_cpus_to_use,
										job_manager=job_manager,
										display_name=f"{self.name} docking",
										protein_file=protein_file,
										pocket_definition=pocket_definition,
										exhaustiveness=exhaustiveness,
										n_poses=n_poses)

			# Process results
			processed_results = []
			for result_file in results:
				if result_file and Path(result_file).exists():
					processed_results.append(self.process_docking_result(result_file, n_poses))

			# Combine results
			combined_results = pd.concat(processed_results, ignore_index=True)

			# Write output SDF if requested
			if output_sdf:
				PandasTools.WriteSDF(combined_results,
					str(output_sdf),
					molColName="Molecule",
					idName="Pose ID",
					properties=list(combined_results.columns))

			return combined_results

		finally:
			self.remove_temp_dir(temp_dir)

	def _create_batches(self, sdf_file: Path, n_cpus: int) -> List[Path]:
		"""
		Creates batches of compounds from an SDF file based on the number of CPUs.
		"""
		temp_dir = self.create_temp_dir()
		batches = []

		with open(sdf_file, "r") as infile:
			sdf_lines = infile.readlines()

		total_compounds = sdf_lines.count("$$$$\n")

		if total_compounds > n_cpus:
			compounds_per_batch = math.ceil(total_compounds / n_cpus)

			compound_count = 0
			batch_index = 1
			current_batch_lines = []

			for line in sdf_lines:
				current_batch_lines.append(line)

				if line.startswith("$$$$"):
					compound_count += 1

					if compound_count % compounds_per_batch == 0:
						batch_file = temp_dir / f"batch_{batch_index}.sdf"
						with open(batch_file, "w") as outfile:
							outfile.writelines(current_batch_lines)
						batches.append(batch_file)
						current_batch_lines = []
						batch_index += 1

			# Write the remaining compounds to the last batch file
			if current_batch_lines:
				batch_file = temp_dir / f"batch_{batch_index}.sdf"
				with open(batch_file, "w") as outfile:
					outfile.writelines(current_batch_lines)
				batches.append(batch_file)
		else:
			# If there are fewer or equal compounds than CPUs, create a single batch
			batch_file = temp_dir / "batch_1.sdf"
			with open(batch_file, "w") as outfile:
				outfile.writelines(sdf_lines)
			batches.append(batch_file)

		return batches

	def dock_batch(self,
		batch_file: Path,
		protein_file: Path,
		pocket_definition: dict,
		exhaustiveness: int,
		n_poses: int) -> Path:
		"""
		Docks a batch of ligands and returns the path to the results file.
		"""
		results_path = self.create_temp_dir() / f"{batch_file.stem}_results.sdf"

		# Implement the docking logic here, using the batch_file instead of a single ligand file
		# This method should be overridden in the specific docking function classes (e.g., SminaDocking)

		return results_path

	@staticmethod
	def combine_results(results: List[pd.DataFrame]) -> pd.DataFrame:
		"""
		Combines multiple DataFrames of docking results into a single DataFrame.
		"""
		return pd.concat(results, ignore_index=True)

	def create_temp_dir(self) -> Path:
		"""
		Creates a temporary directory for the docking function.

		Returns:
			Path: The path to the temporary directory.
		"""
		os.makedirs(Path.home() / "dockm8_temp_files", exist_ok=True)
		temp_dir = Path(Path.home() / "dockm8_temp_files" / f"dockm8_{self.name}_{os.getpid()}")
		temp_dir.mkdir(parents=True, exist_ok=True)
		return temp_dir

	@staticmethod
	def remove_temp_dir(temp_dir: Path):
		"""
		Removes the temporary directory.

		Args:
			temp_dir (Path): The path to the temporary directory.
		"""
		shutil.rmtree(str(temp_dir), ignore_errors=True)
