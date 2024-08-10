import subprocess
import sys
import time
import traceback
from pathlib import Path
import os

import pandas as pd

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog
from scripts.setup.software_manager import ensure_software_installed


class RTMScore(ScoringFunction):

	"""
    RTMScore scoring function implementation.
    """

	@ensure_software_installed("RTMSCORE")
	def __init__(self, software_path: Path):
		super().__init__("RTMScore", "RTMScore", "max", (0, 100), software_path)

	def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
		"""
        Rescore the molecules in the given SDF file using the RTMScore scoring function.

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
			rtmscore_results = Path(temp_dir) / f"{self.column_name}_scores.csv"
			pocket_file = Path(str(protein_file).replace(".pdb", "_pocket.pdb"))

			rtmscore_cmd = (f"cd {self.software_path}/RTMScore-main/example/ &&"
							f" python rtmscore.py"
							f" -p {pocket_file}"
							f" -l {sdf_file}"
							f" -o {rtmscore_results}"
							" -pl"
							f" -m {self.software_path}/RTMScore-main/trained_models/rtmscore_model1.pth")

			try:
				subprocess.run(rtmscore_cmd,
								shell=True,
								check=True,
								stdout=subprocess.DEVNULL,
								stderr=subprocess.DEVNULL)
			except subprocess.CalledProcessError as e:
				if not os.path.exists(os.path.join(self.software_path, "RTMScore-main", "example", "rtmscore.py")):
					printlog(
						"ERROR: Failed to run RTMScore! The software folder does not contain rtmscore.py, please reinstall RTMScore."
					)
				else:
					printlog(
						"ERROR: Failed to run RTMScore! This was likely caused by a failure in generating the pocket graph."
					)
				printlog(traceback.format_exc())
				return pd.DataFrame()

			rtmscore_results_df = pd.read_csv(rtmscore_results)
			rtmscore_results_df = rtmscore_results_df.rename(columns={"id": "Pose ID", "score": self.column_name})
			rtmscore_results_df["Pose ID"] = rtmscore_results_df["Pose ID"].str.rsplit("-", n=1).str[0]
			rtmscore_results_df = rtmscore_results_df[["Pose ID", self.column_name]]

			end_time = time.perf_counter()
			printlog(f"Rescoring with RTMScore complete in {end_time - start_time:.4f} seconds!")
			return rtmscore_results_df

		except Exception as e:
			printlog(f"ERROR: An unexpected error occurred during RTMScore rescoring:")
			printlog(traceback.format_exc())
			return pd.DataFrame()
		finally:
			self.remove_temp_dir(temp_dir)
