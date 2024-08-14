import subprocess
import sys
import time
import traceback
from pathlib import Path
import os
from typing import List

import pandas as pd
from rdkit.Chem import PandasTools

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf_single_str
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.parallel_executor import parallel_executor
from scripts.setup.software_manager import ensure_software_installed


class DLIGAND2(ScoringFunction):

	"""
    DLIGAND2 scoring function implementation.
    """

	def __init__(self, software_path: Path):
		super().__init__("DLIGAND2", "DLIGAND2", "min", (-200, 100), software_path)
		self.software_path = software_path
		ensure_software_installed("DLIGAND2", software_path)

	def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
		"""
        Rescore the molecules in the given SDF file using the DLIGAND2 scoring function.

        Args:
            sdf_file (str): The path to the SDF file.
            n_cpus (int): The number of CPUs to use for parallel processing.
            protein_file (str): The path to the protein file.
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: A DataFrame containing the rescored molecules.
        """
		start_time = time.perf_counter()

		temp_dir = self.create_temp_dir()
		try:
			split_files_folder = split_sdf_single_str(Path(temp_dir), sdf_file)
			split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

			rescoring_results = parallel_executor(self._rescore_split_file,
				split_files_sdfs,
				n_cpus,
				display_name=self.name,
				protein_file=protein_file)

			dligand2_rescoring_results = self._combine_rescoring_results(rescoring_results)

			end_time = time.perf_counter()
			printlog(f"Rescoring with DLIGAND2 complete in {end_time - start_time:.4f} seconds!")
			return dligand2_rescoring_results
		except Exception as e:
			printlog(f"ERROR: An unexpected error occurred during DLIGAND2 rescoring:")
			printlog(traceback.format_exc())
			return pd.DataFrame()
		finally:
			self.remove_temp_dir(temp_dir)

	def _rescore_split_file(self, split_file: Path, protein_file: str) -> Path:
		"""
        Rescore a single split SDF file.

        Args:
            split_file (Path): The path to the split SDF file.
            protein_file (str): The path to the protein file.

        Returns:
            Path: The path to the rescored CSV file.
        """
		try:
			df = PandasTools.LoadSDF(str(split_file), idName="Pose ID", molColName=None)
			df = df[["Pose ID"]]

			mol2_file = split_file.with_suffix('.mol2')
			convert_molecules(split_file, mol2_file, "sdf", "mol2")

			dligand2_cmd = (f"cd {self.software_path}/DLIGAND2/bin &&"
				f" ./dligand2.gnu -etype 2"
				f" -P {protein_file}"
				f" -L {mol2_file}")

			result = subprocess.run(dligand2_cmd, shell=True, capture_output=True, text=True)

			try:
				energy = round(float(result.stdout.strip()), 2)
			except (ValueError, IndexError):
				printlog(f"Warning: Could not parse energy for file {split_file}. Setting to None.")
				energy = None

			df[self.column_name] = [energy]
			output_csv = split_file.parent / f"{split_file.stem}_score.csv"
			df.to_csv(output_csv, index=False)
			return output_csv
		except Exception as e:
			printlog(f"DLIGAND2 rescoring failed for {split_file}:")
			printlog(traceback.format_exc())
			return None

	def _combine_rescoring_results(self, result_files: List[Path]) -> pd.DataFrame:
		"""
        Combine rescoring results from multiple CSV files.

        Args:
            result_files (List[Path]): List of paths to rescored CSV files.

        Returns:
            pd.DataFrame: Combined DataFrame with rescoring results.
        """
		try:
			dataframes = []
			for file in result_files:
				if file and file.is_file():
					df = pd.read_csv(file)
					dataframes.append(df)

			if not dataframes:
				printlog("No valid CSV files found with DLIGAND2 scores.")
				return pd.DataFrame()

			combined_results = pd.concat(dataframes, ignore_index=True)
			return combined_results[["Pose ID", self.column_name]]
		except Exception as e:
			printlog(f"ERROR: Could not combine {self.column_name} rescored poses")
			printlog(traceback.format_exc())
			return pd.DataFrame()
