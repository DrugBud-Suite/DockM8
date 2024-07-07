import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Assuming similar path structure as in the provided code
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.utilities import delete_files


class GenScore(ScoringFunction):

	def __init__(self, score_type):
		if score_type == "scoring":
			super().__init__("GenScore-scoring", "GenScore-scoring", "max", (0, 200))
			self.model = "../trained_models/GatedGCN_ft_1.0_1.pth"
			self.encoder = "gatedgcn"
		elif score_type == "docking":
			super().__init__("GenScore-docking", "GenScore-docking", "max", (0, 200))
			self.model = "../trained_models/GT_0.0_1.pth"
			self.encoder = "gt"
		elif score_type == "balanced":
			super().__init__("GenScore-balanced", "GenScore-balanced", "max", (0, 200))
			self.model = "../trained_models/GT_ft_0.5_1.pth"
			self.encoder = "gt"
		else:
			raise ValueError(f"Invalid GenScore type: {score_type}")

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		rescoring_folder = kwargs.get("rescoring_folder")
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")
		pocket_file = str(protein_file).replace(".pdb", "_pocket.pdb")

		(rescoring_folder / f"{self.column_name}_rescoring").mkdir(parents=True, exist_ok=True)

		split_files_folder = split_sdf_str(rescoring_folder / f"{self.column_name}_rescoring", sdf, n_cpus)
		split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]
		global genscore_rescoring_splitted
		def genscore_rescoring_splitted(split_file):
			try:
				cmd = (f"cd {software}/GenScore/example/ &&"
					"conda run -n genscore python genscore.py"
					f" -p {pocket_file}"
					f" -l {split_file}"
					f" -o {rescoring_folder / f'{self.column_name}_rescoring' / split_file.stem}"
					f" -m {self.model}"
					f" -e {self.encoder}")

				subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

				return rescoring_folder / f"{self.column_name}_rescoring" / f"{split_file.stem}.csv"
			except Exception as e:
				printlog(f"Error occurred while running GenScore on {split_file}: {e}")
				return None

		rescoring_results = parallel_executor(genscore_rescoring_splitted, split_files_sdfs, n_cpus)

		genscore_dataframes = []
		for result_file in rescoring_results:
			if result_file and Path(result_file).is_file():
				df = pd.read_csv(result_file)
				genscore_dataframes.append(df)
				os.remove(result_file)

		if not genscore_dataframes:
			printlog(f"ERROR: No valid results found for {self.column_name} rescoring!")
			return pd.DataFrame()

		genscore_rescoring_results = pd.concat(genscore_dataframes)
		genscore_rescoring_results.rename(columns={"id": "Pose ID", "score": self.column_name}, inplace=True)
		genscore_scores_path = rescoring_folder / f"{self.column_name}_rescoring" / f"{self.column_name}_scores.csv"
		genscore_rescoring_results.to_csv(genscore_scores_path, index=False)

		delete_files(rescoring_folder / f"{self.column_name}_rescoring", f"{self.column_name}_scores.csv")

		toc = time.perf_counter()
		printlog(f"Rescoring with {self.column_name} complete in {toc - tic:0.4f}!")
		return genscore_rescoring_results
