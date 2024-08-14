import subprocess
import sys
import time
import traceback
from pathlib import Path
import os
from typing import List

import pandas as pd

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.pocket_finding.utils import extract_pocket
from scripts.setup.software_manager import ensure_software_installed


class GenScore(ScoringFunction):

	"""
    GenScore scoring function implementation.
    """

	def __init__(self, score_type: str, software_path: Path):
		genscore_path = software_path / "GenScore"
		self.software_path = software_path
		ensure_software_installed("GenScore", software_path)
		if score_type == "scoring":
			super().__init__("GenScore-scoring", "GenScore-scoring", "max", (0, 200), genscore_path)
			self.model = genscore_path / "trained_models" / "GatedGCN_ft_1.0_1.pth"
			self.encoder = "gatedgcn"
		elif score_type == "docking":
			super().__init__("GenScore-docking", "GenScore-docking", "max", (0, 200), genscore_path)
			self.model = genscore_path / "trained_models" / "GT_0.0_1.pth"
			self.encoder = "gt"
		elif score_type == "balanced":
			super().__init__("GenScore-balanced", "GenScore-balanced", "max", (0, 200), genscore_path)
			self.model = genscore_path / "trained_models" / "GT_ft_0.5_1.pth"
			self.encoder = "gt"
		else:
			raise ValueError(f"Invalid GenScore type: {score_type}")



	def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
		"""
        Rescore the molecules in the given SDF file using the GenScore scoring function.

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
			pocket_file = Path(str(protein_file).replace(".pdb", "_pocket.pdb"))
			if not pocket_file.is_file():
				pocket_file = extract_pocket(kwargs.get("pocket_definition"), protein_file)

			split_files_folder = split_sdf_str(Path(temp_dir), sdf_file, n_cpus)
			split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

			rescoring_results = parallel_executor(self._rescore_split_file,
				split_files_sdfs,
				n_cpus,
				display_name=self.name,
				pocket_file=pocket_file)

			genscore_rescoring_results = self._combine_rescoring_results(rescoring_results)

			end_time = time.perf_counter()
			printlog(f"Rescoring with {self.name} complete in {end_time - start_time:.4f} seconds!")
			return genscore_rescoring_results
		except Exception as e:
			printlog(f"ERROR: An unexpected error occurred during {self.name} rescoring:")
			printlog(traceback.format_exc())
			return pd.DataFrame()
		finally:
			self.remove_temp_dir(temp_dir)

	def _rescore_split_file(self, split_file: Path, pocket_file: Path) -> Path:
		"""
        Rescore a single split SDF file.

        Args:
            split_file (Path): The path to the split SDF file.
            pocket_file (Path): The path to the protein pocket file.

        Returns:
            Path: The path to the rescored CSV file.
        """
		try:
			genscore_cmd = (f"cd {self.software_path}/example/ &&"
				f" conda run -n genscore python genscore.py"
				f" -p {pocket_file}"
				f" -l {split_file}"
				f" -o {split_file.parent / split_file.stem}"
				f" -m {self.model}"
				f" -e {self.encoder}")

			subprocess.run(genscore_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			return split_file.parent / f"{split_file.stem}.csv"
		except subprocess.CalledProcessError as e:
			printlog(f"GenScore rescoring failed for {split_file}:")
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
				printlog(f"No valid CSV files found with {self.name} scores.")
				return pd.DataFrame()

			combined_results = pd.concat(dataframes, ignore_index=True)
			combined_results.rename(columns={"id": "Pose ID", "score": self.column_name}, inplace=True)
			combined_results["Pose ID"] = combined_results["Pose ID"].apply(lambda x: x.split("-")[0])
			return combined_results[["Pose ID", self.column_name]]
		except Exception as e:
			printlog(f"ERROR: Could not combine {self.column_name} rescored poses")
			printlog(traceback.format_exc())
			return pd.DataFrame()
