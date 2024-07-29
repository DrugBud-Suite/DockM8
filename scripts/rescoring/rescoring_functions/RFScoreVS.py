import os
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
from scripts.setup.software_manager import ensure_software_installed

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class RFScoreVS(ScoringFunction):

	@ensure_software_installed("RF_SCORE_VS")
	def __init__(self, software_path: Path):
		super().__init__("RFScoreVS", "RFScoreVS", "max", (5, 10), software_path)

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		temp_dir = self.create_temp_dir()
		try:
			split_files_folder = split_sdf_str(Path(temp_dir), sdf, n_cpus)
			split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

			global rf_score_vs_splitted

			def rf_score_vs_splitted(split_file, protein_file):
				rfscorevs_cmd = f"{software}/rf-score-vs --receptor {protein_file} {split_file} -O {Path(temp_dir) / Path(split_file).stem}_RFScoreVS_scores.csv -n 1"
				subprocess.call(rfscorevs_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				return

			parallel_executor(rf_score_vs_splitted,
					split_files_sdfs,
					n_cpus,
					display_name=self.column_name,
					protein_file=protein_file)

			try:
				rfscorevs_dataframes = [
					pd.read_csv(Path(temp_dir) / file, delimiter=",", header=0)
					for file in os.listdir(temp_dir)
					if file.startswith("split") and file.endswith(".csv")]
				rfscorevs_results = pd.concat(rfscorevs_dataframes)
				rfscorevs_results.rename(columns={"name": "Pose ID", "RFScoreVS_v2": self.column_name}, inplace=True)
			except Exception as e:
				printlog("ERROR: Failed to process RFScoreVS results!")
				printlog(e)
				return pd.DataFrame()

			toc = time.perf_counter()
			printlog(f"Rescoring with RFScoreVS complete in {toc-tic:0.4f}!")
			return rfscorevs_results
		finally:
			self.remove_temp_dir(temp_dir)


# Usage:
# rfscorevs = RFScoreVS()
# results = rfscorevs.rescore(sdf_file, n_cpus, software=software_path, protein_file=protein_file_path)
