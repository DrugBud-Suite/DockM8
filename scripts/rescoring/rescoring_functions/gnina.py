import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import pandas as pd
from rdkit.Chem import PandasTools

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


class Gnina(ScoringFunction):

	def __init__(self, score_type):
		if score_type == "affinity":
			super().__init__("GNINA-Affinity", "GNINA-Affinity", "min", (100, -100))
		elif score_type == "cnn_score":
			super().__init__("CNN-Score", "CNN-Score", "max", (0, 1))
		elif score_type == "cnn_affinity":
			super().__init__("CNN-Affinity", "CNN-Affinity", "max", (0, 20))
		else:
			raise ValueError("Invalid score type for Gnina")
		self.score_type = score_type

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")
		cnn = "crossdock_default2018"

		with tempfile.TemporaryDirectory() as temp_dir:
			split_files = split_sdf_str(Path(temp_dir), sdf, n_cpus)
			split_files_sdfs = [Path(temp_dir) / f for f in os.listdir(split_files) if f.endswith(".sdf")]

			global gnina_rescoring_splitted

			def gnina_rescoring_splitted(split_file, protein_file):
				results = Path(temp_dir) / f"{split_file.stem}_{self.column_name}.sdf"
				gnina_cmd = (f"{software}/gnina"
								f" --receptor {protein_file}"
								f" --ligand {split_file}"
								f" --out {results}"
								" --cpu 1"
								" --score_only"
								f" --cnn {cnn} --no_gpu")
				try:
					subprocess.call(gnina_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				except Exception as e:
					printlog(f"{self.column_name} rescoring failed: " + str(e))
				return results

			rescoring_results = parallel_executor(gnina_rescoring_splitted,
													split_files_sdfs,
													n_cpus,
													protein_file=protein_file)

			gnina_dataframes = [
				PandasTools.LoadSDF(str(file),
									idName="Pose ID",
									molColName=None,
									includeFingerprints=False,
									embedProps=False) for file in rescoring_results if file.is_file()]

		if not gnina_dataframes:
			printlog(f"ERROR: No valid results found for {self.column_name} rescoring!")
			return pd.DataFrame()

		gnina_rescoring_results = pd.concat(gnina_dataframes)
		gnina_rescoring_results.rename(columns={
			"minimizedAffinity": "GNINA-Affinity", "CNNscore": "CNN-Score", "CNNaffinity": "CNN-Affinity"},
										inplace=True)

		gnina_rescoring_results = gnina_rescoring_results[["Pose ID", self.column_name]]

		toc = time.perf_counter()
		printlog(f"Rescoring with {self.column_name} complete in {toc - tic:0.4f}!")
		return gnina_rescoring_results
