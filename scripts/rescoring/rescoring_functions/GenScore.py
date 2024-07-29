import subprocess
import sys
import time
from pathlib import Path
import os
import pandas as pd

# Assuming similar path structure as in the provided code
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.pocket_extraction import extract_pocket
from scripts.setup.software_manager import ensure_software_installed


class GenScore(ScoringFunction):

	@ensure_software_installed("GENSCORE")
	def __init__(self, score_type, software_path: Path):
		if score_type == "scoring":
			super().__init__("GenScore-scoring", "GenScore-scoring", "max", (0, 200), software_path)
			self.model = "../trained_models/GatedGCN_ft_1.0_1.pth"
			self.encoder = "gatedgcn"
			self.software_path = software_path
		elif score_type == "docking":
			super().__init__("GenScore-docking", "GenScore-docking", "max", (0, 200), software_path)
			self.model = "../trained_models/GT_0.0_1.pth"
			self.encoder = "gt"
			self.software_path = software_path
		elif score_type == "balanced":
			super().__init__("GenScore-balanced", "GenScore-balanced", "max", (0, 200), software_path)
			self.model = "../trained_models/GT_ft_0.5_1.pth"
			self.encoder = "gt"
			self.software_path = software_path
		else:
			raise ValueError(f"Invalid GenScore type: {score_type}")

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		protein_file = kwargs.get("protein_file")
		pocket_file = str(protein_file).replace(".pdb", "_pocket.pdb")

		if not Path(pocket_file).is_file():
			pocket_file = extract_pocket(kwargs.get('pocket_definition'), pocket_file)

		temp_dir = self.create_temp_dir()
		try:
			split_files_folder = split_sdf_str(Path(temp_dir), sdf, n_cpus)
			split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

			global genscore_rescoring_splitted

			def genscore_rescoring_splitted(split_file):
				try:
					cmd = (f"cd {self.software_path}/GenScore/example/ &&"
						"conda run -n genscore python genscore.py"
						f" -p {pocket_file}"
						f" -l {split_file}"
						f" -o {Path(temp_dir) / Path(split_file).stem}"
						f" -m {self.model}"
						f" -e {self.encoder}")

					subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

					return Path(temp_dir) / f"{Path(split_file).stem}.csv"
				except Exception as e:
					printlog(f"Error occurred while running GenScore on {split_file}: {e}")
					return None

			rescoring_results = parallel_executor(genscore_rescoring_splitted,
				split_files_sdfs,
				n_cpus,
				display_name=self.column_name)

			genscore_dataframes = []
			for result_file in rescoring_results:
				if result_file and Path(result_file).is_file():
					df = pd.read_csv(result_file)
					genscore_dataframes.append(df)

			if not genscore_dataframes:
				printlog(f"ERROR: No valid results found for {self.column_name} rescoring!")
				return pd.DataFrame()

			genscore_rescoring_results = pd.concat(genscore_dataframes)
			genscore_rescoring_results.rename(columns={"id": "Pose ID", "score": self.column_name}, inplace=True)

			toc = time.perf_counter()
			printlog(f"Rescoring with {self.column_name} complete in {toc - tic:0.4f}!")
			return genscore_rescoring_results
		finally:
			self.remove_temp_dir(temp_dir)


# Usage:
# genscore = GenScore("scoring")  # or "docking" or "balanced"
# results = genscore.rescore(sdf_file, n_cpus, protein_file=protein_file_path)
