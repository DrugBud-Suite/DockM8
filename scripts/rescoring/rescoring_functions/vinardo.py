import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.utilities import delete_files

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Vinardo(ScoringFunction):

	def __init__(self):
		super().__init__("Vinardo", "Vinardo", "min", (200, 20))

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> DataFrame:
		tic = time.perf_counter()

		rescoring_folder = kwargs.get("rescoring_folder")
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		split_files_folder = split_sdf_str(rescoring_folder / f"{self.column_name}_rescoring", sdf, n_cpus)
		split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

		vinardo_rescoring_folder = rescoring_folder / f"{self.column_name}_rescoring"
		vinardo_rescoring_folder.mkdir(parents=True, exist_ok=True)
		global vinardo_rescoring_splitted
		def vinardo_rescoring_splitted(split_file, protein_file):
			vinardo_rescoring_folder = rescoring_folder / f"{self.column_name}_rescoring"
			results = vinardo_rescoring_folder / f"{Path(split_file).stem}_{self.column_name}.sdf"
			vinardo_cmd = (f"{software}/gnina"
				f" --receptor {protein_file}"
				f" --ligand {split_file}"
				f" --out {results}"
				" --score_only"
				" --scoring vinardo"
				" --cnn_scoring none")
			try:
				subprocess.call(vinardo_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			except Exception as e:
				printlog(f"{self.column_name} rescoring failed: " + str(e))
			return

		parallel_executor(vinardo_rescoring_splitted, split_files_sdfs, n_cpus, protein_file=protein_file)

		try:
			vinardo_dataframes = [
				PandasTools.LoadSDF(str(rescoring_folder / f"{self.column_name}_rescoring" / file),
					idName="Pose ID",
					molColName=None,
					includeFingerprints=False,
					embedProps=False)
				for file in os.listdir(rescoring_folder / f"{self.column_name}_rescoring")
				if file.startswith("split") and file.endswith(".sdf")]
		except Exception as e:
			printlog(f"ERROR: Failed to Load {self.column_name} rescoring SDF file!")
			printlog(e)
			return pd.DataFrame()

		try:
			vinardo_rescoring_results = pd.concat(vinardo_dataframes)
		except Exception as e:
			printlog(f"ERROR: Could not combine {self.column_name} rescored poses")
			printlog(e)
			return pd.DataFrame()

		vinardo_rescoring_results.rename(columns={"minimizedAffinity": self.column_name}, inplace=True)
		vinardo_rescoring_results = vinardo_rescoring_results[["Pose ID", self.column_name]]
		vinardo_scores_path = vinardo_rescoring_folder / f"{self.column_name}_scores.csv"
		vinardo_rescoring_results.to_csv(vinardo_scores_path, index=False)
		delete_files(vinardo_rescoring_folder, f"{self.column_name}_scores.csv")
		toc = time.perf_counter()
		printlog(f"Rescoring with Vinardo complete in {toc - tic:0.4f}!")
		return vinardo_rescoring_results
