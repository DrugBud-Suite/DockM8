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


class AAScore(ScoringFunction):

	def __init__(self):
		super().__init__("AAScore", "AAScore", "max", (100, -100))

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		pocket = str(protein_file).replace(".pdb", "_pocket.pdb")

		if n_cpus == 1:
			with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as temp_file:
				AAscore_cmd = f"python {software}/AA-Score-Tool-main/AA_Score.py --Rec {pocket} --Lig {sdf} --Out {temp_file.name}"
				subprocess.call(AAscore_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				AAScore_rescoring_results = pd.read_csv(temp_file.name,
														delimiter="\t",
														header=None,
														names=["Pose ID", self.column_name])
			os.unlink(temp_file.name)
		else:
			with tempfile.TemporaryDirectory() as temp_dir:
				split_files = split_sdf_str(Path(temp_dir), sdf, n_cpus)
				split_files_sdfs = [Path(temp_dir) / f for f in os.listdir(split_files) if f.endswith(".sdf")]

				global AAScore_rescoring_splitted

				def AAScore_rescoring_splitted(split_file):
					results = Path(temp_dir) / f"{split_file.stem}_AAScore.csv"
					AAScore_cmd = f"python {software}/AA-Score-Tool-main/AA_Score.py --Rec {pocket} --Lig {split_file} --Out {results}"
					subprocess.call(AAScore_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

				parallel_executor(AAScore_rescoring_splitted, split_files_sdfs, n_cpus)

				AAScore_dataframes = [
					pd.read_csv(Path(temp_dir) / file, delimiter="\t", header=None, names=["Pose ID", self.column_name])
					for file in os.listdir(temp_dir)
					if file.endswith("_AAScore.csv")]
				AAScore_rescoring_results = pd.concat(AAScore_dataframes)

		toc = time.perf_counter()
		printlog(f"Rescoring with AAScore complete in {toc - tic:0.4f}!")
		return AAScore_rescoring_results
