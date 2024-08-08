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
from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.setup.software_manager import ensure_software_installed


class LinF9(ScoringFunction):

	"""
    LinF9 scoring function implementation.
    """

	@ensure_software_installed("LIN_F9")
	def __init__(self, software_path: Path):
		super().__init__("LinF9", "LinF9", "min", (100, -100), software_path)

	def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
		"""
        Rescore the molecules in the given SDF file using the LinF9 scoring function.

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
			split_files_folder = split_sdf_str(Path(temp_dir), sdf_file, n_cpus)
			split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

			rescoring_results = parallel_executor(self._rescore_split_file,
													split_files_sdfs,
													n_cpus,
													display_name=self.name,
													protein_file=protein_file)

			linf9_dataframes = self._load_rescoring_results(rescoring_results)
			linf9_rescoring_results = self._combine_rescoring_results(linf9_dataframes)

			end_time = time.perf_counter()
			printlog(f"Rescoring with LinF9 complete in {end_time - start_time:.4f} seconds!")
			return linf9_rescoring_results
		except Exception as e:
			printlog(f"ERROR: An unexpected error occurred during LinF9 rescoring:")
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
            Path: The path to the rescored SDF file.
        """
		results = split_file.parent / f"{split_file.stem}_LinF9.sdf"
		linf9_cmd = (f"{self.software_path}/LinF9"
						f" --receptor {protein_file}"
						f" --ligand {split_file}"
						f" --out {results}"
						" --cpu 1"
						" --scoring Lin_F9 --score_only")
		try:
			subprocess.run(linf9_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		except subprocess.CalledProcessError as e:
			printlog(f"LinF9 rescoring failed for {split_file}:")
			printlog(traceback.format_exc())
		return results

	def _load_rescoring_results(self, result_files: List[Path]) -> List[pd.DataFrame]:
		"""
        Load rescoring results from SDF files.

        Args:
            result_files (List[Path]): List of paths to rescored SDF files.

        Returns:
            List[pd.DataFrame]: List of DataFrames containing the rescoring results.
        """
		dataframes = []
		for file in result_files:
			try:
				df = PandasTools.LoadSDF(str(file),
											idName="Pose ID",
											molColName=None,
											includeFingerprints=False,
											embedProps=False)
				dataframes.append(df)
			except Exception as e:
				printlog(f"ERROR: Failed to Load {self.column_name} rescoring SDF file: {file}")
				printlog(traceback.format_exc())
		return dataframes

	def _combine_rescoring_results(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
		"""
        Combine rescoring results from multiple DataFrames.

        Args:
            dataframes (List[pd.DataFrame]): List of DataFrames containing rescoring results.

        Returns:
            pd.DataFrame: Combined DataFrame with rescoring results.
        """
		try:
			combined_results = pd.concat(dataframes, ignore_index=True)
			combined_results.rename(columns={"minimizedAffinity": self.column_name}, inplace=True)
			return combined_results[["Pose ID", self.column_name]]
		except Exception as e:
			printlog(f"ERROR: Could not combine {self.column_name} rescored poses")
			printlog(traceback.format_exc())
			return pd.DataFrame()
