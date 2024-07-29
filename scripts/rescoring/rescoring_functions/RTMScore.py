import subprocess
import sys
import time
import os
import warnings
from pathlib import Path

import pandas as pd

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog
from scripts.setup.software_manager import ensure_software_installed

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class RTMScore(ScoringFunction):

	@ensure_software_installed("RTMSCORE")
	def __init__(self, software_path: Path):
		super().__init__("RTMScore", "RTMScore", "max", (0, 100), software_path)
		self.software_path = software_path

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		temp_dir = self.create_temp_dir()
		try:
			RTMScore_rescoring_results = Path(temp_dir) / f"{self.column_name}_scores.csv"
			try:
				RTMScore_command = (f'cd {temp_dir} && python {software}/RTMScore-main/example/rtmscore.py' +
					f' -p {str(protein_file).replace(".pdb", "_pocket.pdb")}' + f" -l {sdf}" +
					" -o RTMScore_scores" + " -pl"
					f" -m {software}/RTMScore-main/trained_models/rtmscore_model1.pth")
				subprocess.call(RTMScore_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			except Exception as e:
				if not os.path.exists(os.path.join(software, "RTMScore-main", "example", "rtmscore.py")):
					printlog(
						"ERROR: Failed to run RTMScore! The software folder does not contain rtmscore.py, please reinstall RTMScore."
					)
				else:
					printlog(
						f"ERROR: Failed to run RTMScore! This was likely caused by a failure in generating the pocket graph : {e}."
					)
				return pd.DataFrame()

			df = pd.read_csv(RTMScore_rescoring_results)
			df = df.rename(columns={"id": "Pose ID", "score": f"{self.column_name}"})
			df["Pose ID"] = df["Pose ID"].str.rsplit("-", n=1).str[0]

			toc = time.perf_counter()
			printlog(f"Rescoring with RTMScore complete in {toc-tic:0.4f}!")
			return df
		finally:
			self.remove_temp_dir(temp_dir)


# Usage:
# rtmscore = RTMScore()
# results = rtmscore.rescore(sdf_file, n_cpus, software=software_path, protein_file=protein_file_path)
