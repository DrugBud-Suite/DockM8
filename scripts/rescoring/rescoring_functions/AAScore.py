import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import List

import pandas as pd

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.setup.software_manager import ensure_software_installed


class AAScore(ScoringFunction):

	"""
    AAScore scoring function implementation.
    """

	@ensure_software_installed("AA_SCORE")
	def __init__(self, software_path: Path):
		super().__init__("AAScore", "AAScore", "max", (100, -100), software_path)

	def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
		"""
        Rescore the molecules in the given SDF file using the AAScore scoring function.

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
			pocket_file = str(protein_file).replace(".pdb", "_pocket.pdb")

			if n_cpus == 1:
				aascore_rescoring_results = self._rescore_single_process(sdf_file, pocket_file, temp_dir)
			else:
				aascore_rescoring_results = self._rescore_multi_process(sdf_file, pocket_file, n_cpus, temp_dir)

			end_time = time.perf_counter()
			printlog(f"Rescoring with AAScore complete in {end_time - start_time:.4f} seconds!")
			return aascore_rescoring_results
		except Exception as e:
			printlog(f"ERROR: An unexpected error occurred during AAScore rescoring:")
			printlog(traceback.format_exc())
			return pd.DataFrame()
		finally:
			self.remove_temp_dir(temp_dir)

	def _rescore_single_process(self, sdf_file: str, pocket_file: str, temp_dir: Path) -> pd.DataFrame:
		"""
        Rescore using a single process.

        Args:
            sdf_file (str): The path to the SDF file.
            pocket_file (str): The path to the pocket file.
            temp_dir (Path): The temporary directory path.

        Returns:
            pd.DataFrame: A DataFrame containing the rescoring results.
        """
		results = temp_dir / "rescored_AAScore.csv"
		aascore_cmd = (f"conda run -n AAScore {self.software_path}/AA-Score-Tool-main/AA_Score.py"
						f" --Rec {pocket_file}"
						f" --Lig {sdf_file}"
						f" --Out {results}")
		try:
			subprocess.run(aascore_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			return pd.read_csv(results, delimiter="\t", header=None, names=["Pose ID", self.column_name])
		except subprocess.CalledProcessError as e:
			printlog(f"AAScore rescoring failed:")
			printlog(traceback.format_exc())
			return pd.DataFrame()

	def _rescore_multi_process(self, sdf_file: str, pocket_file: str, n_cpus: int, temp_dir: Path) -> pd.DataFrame:
		"""
        Rescore using multiple processes.

        Args:
            sdf_file (str): The path to the SDF file.
            pocket_file (str): The path to the pocket file.
            n_cpus (int): The number of CPUs to use for parallel processing.
            temp_dir (Path): The temporary directory path.

        Returns:
            pd.DataFrame: A DataFrame containing the rescoring results.
        """
		split_files_folder = split_sdf_str(Path(temp_dir), sdf_file, n_cpus)
		split_files_sdfs = list(split_files_folder.glob("*.sdf"))

		rescoring_results = parallel_executor(self._rescore_split_file,
												split_files_sdfs,
												n_cpus,
												display_name=self.name,
												pocket_file=pocket_file)

		return self._combine_rescoring_results(rescoring_results)

	def _rescore_split_file(self, split_file: Path, pocket_file: str) -> Path:
		"""
        Rescore a single split SDF file.

        Args:
            split_file (Path): The path to the split SDF file.
            pocket_file (str): The path to the pocket file.

        Returns:
            Path: The path to the rescored CSV file.
        """
		results = split_file.parent / f"{split_file.stem}_AAScore.csv"
		aascore_cmd = (f"python {self.software_path}/AA-Score-Tool-main/AA_Score.py"
						f" --Rec {pocket_file}"
						f" --Lig {split_file}"
						f" --Out {results}")
		try:
			subprocess.run(aascore_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			return results
		except subprocess.CalledProcessError as e:
			printlog(f"AAScore rescoring failed for {split_file}:")
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
					df = pd.read_csv(file, delimiter="\t", header=None, names=["Pose ID", self.column_name])
					dataframes.append(df)

			if not dataframes:
				printlog("No valid CSV files found with AAScore scores.")
				return pd.DataFrame()

			combined_results = pd.concat(dataframes, ignore_index=True)
			return combined_results[["Pose ID", self.column_name]]
		except Exception as e:
			printlog(f"ERROR: Could not combine {self.column_name} rescored poses")
			printlog(traceback.format_exc())
			return pd.DataFrame()
