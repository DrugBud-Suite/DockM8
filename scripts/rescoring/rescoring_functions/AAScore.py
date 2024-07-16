import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class AAScore(ScoringFunction):

	"""
	AAScore class for performing rescoring using the AAScore scoring function.

	Args:
		ScoringFunction (class): Base class for scoring functions.

	Attributes:
		name (str): Name of the scoring function.
		column_name (str): Name of the column in the result dataframe.
		score_type (str): Type of scoring (e.g., "max", "min").
		score_range (tuple): Range of scores (e.g., (100, -100)).

	Methods:
		rescore(sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
			Rescores the given SDF file using the AAScore scoring function.

	"""

	def __init__(self):
		super().__init__("AAScore", "AAScore", "max", (100, -100))

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		"""
		Rescores the given SDF file using the AAScore scoring function.

		Args:
			sdf (str): Path to the SDF file.
			n_cpus (int): Number of CPUs to use for parallel execution.
			**kwargs: Additional keyword arguments.

		Returns:
			pd.DataFrame: DataFrame containing the rescoring results.

		Raises:
			RuntimeError: If the AAScore rescoring fails.
			ValueError: If there is an error in AAScore rescoring.

		"""
		tic = time.perf_counter()
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		temp_dir = self.create_temp_dir()
		try:
			pocket = str(protein_file).replace(".pdb", "_pocket.pdb")

			if n_cpus == 1:
				results = Path(temp_dir) / "rescored_AAScore.csv"
				AAscore_cmd = f"python {software}/AA-Score-Tool-main/AA_Score.py --Rec {pocket} --Lig {sdf} --Out {results}"
				subprocess.run(AAscore_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				AAScore_rescoring_results = pd.read_csv(results,
														delimiter="\t",
														header=None,
														names=["Pose ID", self.column_name])
			else:
				split_files_folder = split_sdf_str(Path(temp_dir), sdf, n_cpus)
				split_files_sdfs = list(split_files_folder.glob("*.sdf"))

				# Define the function for parallel execution
				global AAScore_rescoring_splitted

				def AAScore_rescoring_splitted(split_file):
					results = Path(temp_dir) / f"{split_file.stem}_AAScore.csv"
					AAscore_cmd = f"python {software}/AA-Score-Tool-main/AA_Score.py --Rec {pocket} --Lig {split_file} --Out {results}"
					subprocess.run(AAscore_cmd,
									shell=True,
									check=True,
									stdout=subprocess.DEVNULL,
									stderr=subprocess.STDOUT)

				# Execute the rescoring in parallel
				parallel_executor(AAScore_rescoring_splitted, split_files_sdfs, n_cpus, display_name=self.column_name)

				# Combine results
				result_files = list(Path(temp_dir).glob("*_AAScore.csv"))
				AAScore_dataframes = [
					pd.read_csv(file, delimiter="\t", header=None, names=["Pose ID", self.column_name])
					for file in result_files]
				AAScore_rescoring_results = pd.concat(AAScore_dataframes, ignore_index=True)

			toc = time.perf_counter()
			printlog(f"Rescoring with AAScore complete in {toc - tic:0.4f}!")
			return AAScore_rescoring_results
		except subprocess.CalledProcessError as e:
			raise RuntimeError(f"AAScore rescoring failed: {e}")
		except Exception as e:
			raise ValueError(f"Error in AAScore rescoring: {e}")
		finally:
			self.remove_temp_dir(temp_dir)


# Usage:
# aascore = AAScore()
# results = aascore.rescore(sdf_file, n_cpus, software=software_path, protein_file=protein_file_path)
