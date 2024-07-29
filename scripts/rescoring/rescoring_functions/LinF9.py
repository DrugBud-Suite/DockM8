import os
import subprocess
import sys
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
from scripts.setup.software_manager import ensure_software_installed

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class LinF9(ScoringFunction):

	@ensure_software_installed("LINF9")
	def __init__(self, software_path: Path):
		super().__init__("LinF9", "LinF9", "min", (100, -100), software_path)

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		temp_dir = self.create_temp_dir()
		try:
			split_files_folder = split_sdf_str(Path(temp_dir), sdf, n_cpus)
			split_files_sdfs = [
				Path(split_files_folder) / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

			global LinF9_rescoring_splitted

			def LinF9_rescoring_splitted(split_file, protein_file):
				results = Path(temp_dir) / f"{split_file.stem}_LinF9.sdf"
				LinF9_cmd = (f"{software}/LinF9" + f" --receptor {protein_file}" + f" --ligand {split_file}" +
					f" --out {results}" + " --cpu 1" + " --scoring Lin_F9 --score_only")
				try:
					subprocess.call(LinF9_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				except Exception as e:
					printlog(f"LinF9 rescoring failed: {e}")
				return

			parallel_executor(LinF9_rescoring_splitted,
					split_files_sdfs,
					n_cpus,
					display_name=self.column_name,
					protein_file=protein_file)

			try:
				LinF9_dataframes = [
					PandasTools.LoadSDF(str(Path(temp_dir) / file),
					idName="Pose ID",
					molColName=None,
					includeFingerprints=False,
					embedProps=False)
					for file in os.listdir(temp_dir)
					if file.startswith("split") and file.endswith("_LinF9.sdf")]
			except Exception as e:
				printlog("ERROR: Failed to Load LinF9 rescoring SDF file!")
				printlog(e)
				return pd.DataFrame()

			try:
				LinF9_rescoring_results = pd.concat(LinF9_dataframes)
			except Exception as e:
				printlog("ERROR: Could not combine LinF9 rescored poses")
				printlog(e)
				return pd.DataFrame()

			LinF9_rescoring_results.rename(columns={"minimizedAffinity": self.column_name}, inplace=True)
			LinF9_rescoring_results = LinF9_rescoring_results[["Pose ID", self.column_name]]

			toc = time.perf_counter()
			printlog(f"Rescoring with LinF9 complete in {toc-tic:0.4f}!")
			return LinF9_rescoring_results
		finally:
			self.remove_temp_dir(temp_dir)


# Usage:
# linf9 = LinF9()
# results = linf9.rescore(sdf_file, n_cpus, software=software_path, protein_file=protein_file_path)
