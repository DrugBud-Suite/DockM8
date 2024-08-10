import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
from rdkit.Chem import PandasTools

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog


class PLECScore(ScoringFunction):

	"""
    PLECScore scoring function implementation.
    """

	def __init__(self, software_path: Path):
		super().__init__("PLECScore", "PLECScore", "max", (0, 20), software_path)

	def rescore(self, sdf_file: str, n_cpus: int, protein_file: str, **kwargs) -> pd.DataFrame:
		"""
        Rescore the molecules in the given SDF file using the PLECScore scoring function.

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
			pickle_path = self.software_path / "models/PLECnn_p5_l1_pdbbind2016_s65536.pickle"
			results = Path(temp_dir) / "rescored_PLECnn.sdf"

			plecscore_cmd = (f"oddt_cli {sdf_file}"
								f" --receptor {protein_file}"
								f" -n {n_cpus}"
								f" --score_file {pickle_path}"
								f" -O {results}")

			try:
				subprocess.run(plecscore_cmd,
								shell=True,
								check=True,
								stdout=subprocess.DEVNULL,
								stderr=subprocess.DEVNULL)
			except subprocess.CalledProcessError as e:
				printlog(f"ERROR: PLECScore rescoring failed:")
				printlog(traceback.format_exc())
				return pd.DataFrame()

			plecscore_results_df = PandasTools.LoadSDF(str(results),
														idName="Pose ID",
														molColName=None,
														includeFingerprints=False,
														removeHs=False)

			plecscore_results_df.rename(columns={"PLECnn_p5_l1_s65536": self.column_name}, inplace=True)
			plecscore_results_df = plecscore_results_df[["Pose ID", self.column_name]]

			end_time = time.perf_counter()
			printlog(f"Rescoring with PLECScore complete in {end_time - start_time:.4f} seconds!")
			return plecscore_results_df
		except Exception as e:
			printlog(f"ERROR: An unexpected error occurred during PLECScore rescoring:")
			printlog(traceback.format_exc())
			return pd.DataFrame()
		finally:
			self.remove_temp_dir(temp_dir)
