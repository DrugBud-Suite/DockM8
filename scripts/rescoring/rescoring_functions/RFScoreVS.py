import os
import subprocess
import sys
import tempfile
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


class RFScoreVS(ScoringFunction):

	def __init__(self):
		super().__init__("RFScoreVS", "RFScoreVS", "max", (5, 10))

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		with tempfile.TemporaryDirectory() as temp_dir:
			split_files = split_sdf_str(Path(temp_dir), sdf, n_cpus)
			split_files_sdfs = [Path(temp_dir) / f for f in os.listdir(split_files) if f.endswith(".sdf")]

			global rf_score_vs_splitted

			def rf_score_vs_splitted(split_file, protein_file):
				output_file = Path(temp_dir) / f"{split_file.stem}_RFScoreVS_scores.csv"
				rfscorevs_cmd = f"{software}/rf-score-vs --receptor {protein_file} {split_file} -O {output_file} -n 1"
				subprocess.call(rfscorevs_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				return output_file

			rescoring_results = parallel_executor(rf_score_vs_splitted,
													split_files_sdfs,
													n_cpus,
													protein_file=protein_file)

			try:
				rfscorevs_dataframes = [
					pd.read_csv(file, delimiter=",", header=0) for file in rescoring_results if file.is_file()]
				rfscorevs_results = pd.concat(rfscorevs_dataframes)
				rfscorevs_results.rename(columns={"name": "Pose ID", "RFScoreVS_v2": self.column_name}, inplace=True)
			except Exception as e:
				printlog("ERROR: Failed to process RFScoreVS results!")
				printlog(e)
				return pd.DataFrame()

		toc = time.perf_counter()
		printlog(f"Rescoring with RFScoreVS complete in {toc-tic:0.4f}!")
		return rfscorevs_results
