import subprocess
import sys
import time
import traceback
from pathlib import Path
import re
import shutil
from typing import List, Optional

import pandas as pd
from rdkit.Chem import PandasTools

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.setup.software_manager import ensure_software_installed
from scripts.utilities.file_splitting import split_sdf_single_str
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.parallel_executor import parallel_executor


class CENsible(ScoringFunction):

	"""
    CENsible scoring function implementation.
    """

	def __init__(self, software_path: Path):
		super().__init__("CENsible", "CENsible", "max", (0, 20), software_path)
		self.software_path = software_path
		ensure_software_installed("CENSIBLE", software_path)

	def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
		"""
        Rescore the molecules in the given SDF file using the CENsible scoring function.

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
			smina_path = self.find_executable("smina")
			obabel_path = self.find_executable("obabel")

			if smina_path is None or obabel_path is None:
				raise FileNotFoundError("smina or obabel executable not found. Please ensure they're installed.")

			split_files_folder = split_sdf_single_str(Path(temp_dir), sdf_file)
			split_files_sdfs = [split_files_folder / f for f in split_files_folder.glob("*.sdf")]

			rescoring_results = parallel_executor(self._rescore_split_file,
				split_files_sdfs,
				n_cpus,
				display_name=self.name,
				protein_file=protein_file,
				smina_path=smina_path,
				obabel_path=obabel_path)

			censible_rescoring_results = self._combine_rescoring_results(rescoring_results)

			end_time = time.perf_counter()
			printlog(f"Rescoring with CENsible complete in {end_time - start_time:.4f} seconds!")
			return censible_rescoring_results
		except Exception as e:
			printlog(f"ERROR: An unexpected error occurred during CENsible rescoring:")
			printlog(traceback.format_exc())
			return pd.DataFrame()
		finally:
			self.remove_temp_dir(temp_dir)

	def _rescore_split_file(self, split_file: Path, protein_file: str, smina_path: str, obabel_path: str) -> Path:
		"""
        Rescore a single split SDF file.

        Args:
            split_file (Path): The path to the split SDF file.
            protein_file (str): The path to the protein file.
            smina_path (str): The path to the smina executable.
            obabel_path (str): The path to the obabel executable.

        Returns:
            Path: The path to the rescored CSV file.
        """
		try:
			df = PandasTools.LoadSDF(str(split_file), idName="Pose ID", molColName=None)
			df = df[["Pose ID"]]

			pdb_file = split_file.with_suffix('.pdb')
			convert_molecules(split_file, pdb_file, "sdf", "pdb")

			with pdb_file.open('r') as f:
				ligand_pdb_content = f.read()

			censible_cmd = [
				sys.executable,
				self.software_path / "censible/predict.py",
				"--ligpath",
				str(pdb_file),
				"--recpath",
				protein_file,
				"--smina_exec_path",
				smina_path,
				"--obabel_exec_path",
				obabel_path,
				"--use_cpu"]

			result = subprocess.run(censible_cmd, capture_output=True, text=True)

			score = None
			for line in result.stdout.split('\n'):
				if "score:" in line:
					match = re.search(r'score:\s+(-?\d+\.?\d*)', line)
					if match:
						score = float(match.group(1))
						break

			if score is None:
				printlog(f"Warning: Could not parse score for file {split_file}. Setting to None.")

			df[self.column_name] = [score]
			output_csv = split_file.parent / f"{split_file.stem}_score.csv"
			df.to_csv(output_csv, index=False)
			return output_csv
		except Exception as e:
			printlog(f"CENsible rescoring failed for {split_file}:")
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
				printlog("No valid CSV files found with CENsible scores.")
				return pd.DataFrame()

			combined_results = pd.concat(dataframes, ignore_index=True)
			return combined_results[["Pose ID", self.column_name]]
		except Exception as e:
			printlog(f"ERROR: Could not combine {self.column_name} rescored poses")
			printlog(traceback.format_exc())
			return pd.DataFrame()

	@staticmethod
	def find_executable(name: str) -> Optional[str]:
		"""
        Find the executable file with the given name.

        Args:
            name (str): The name of the executable file.

        Returns:
            Optional[str]: The path to the executable file, or None if not found.
        """
		try:
			result = subprocess.run(['which', name], capture_output=True, text=True)
			return result.stdout.strip()
		except subprocess.CalledProcessError:
			try:
				result = subprocess.run(['where', name], capture_output=True, text=True)
				return result.stdout.strip().split('\n')[0]
			except subprocess.CalledProcessError:
				return None
