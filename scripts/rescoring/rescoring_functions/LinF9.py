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


class LinF9(ScoringFunction):

	def __init__(self):
		super().__init__("LinF9", "LinF9", "min", (100, -100))

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		with tempfile.TemporaryDirectory() as temp_dir:
			split_files = split_sdf_str(Path(temp_dir), sdf, n_cpus)
			split_files_sdfs = [Path(temp_dir) / f for f in os.listdir(split_files) if f.endswith(".sdf")]

			global LinF9_rescoring_splitted

			def LinF9_rescoring_splitted(split_file, protein_file):
				results = Path(temp_dir) / f"{split_file.stem}_LinF9.sdf"
				LinF9_cmd = (f"{software}/LinF9" + f" --receptor {protein_file}" + f" --ligand {split_file}" +
								f" --out {results}" + " --cpu 1" + " --scoring Lin_F9 --score_only")
				try:
					subprocess.call(LinF9_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				except Exception as e:
					printlog(f"LinF9 rescoring failed: {e}")
				return results

			rescoring_results = parallel_executor(LinF9_rescoring_splitted,
													split_files_sdfs,
													n_cpus,
													protein_file=protein_file)

			LinF9_dataframes = [
				PandasTools.LoadSDF(str(file),
									idName="Pose ID",
									molColName=None,
									includeFingerprints=False,
									embedProps=False) for file in rescoring_results if file.is_file()]

		if not LinF9_dataframes:
			printlog("ERROR: Failed to Load LinF9 rescoring SDF file!")
			return pd.DataFrame()

		LinF9_rescoring_results = pd.concat(LinF9_dataframes)
		LinF9_rescoring_results.rename(columns={"minimizedAffinity": self.column_name}, inplace=True)
		LinF9_rescoring_results = LinF9_rescoring_results[["Pose ID", self.column_name]]

		toc = time.perf_counter()
		printlog(f"Rescoring with LinF9 complete in {toc-tic:0.4f}!")
		return LinF9_rescoring_results
