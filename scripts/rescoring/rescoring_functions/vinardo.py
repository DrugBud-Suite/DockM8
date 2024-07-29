import subprocess
import sys
import time
import warnings
from pathlib import Path
import os

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


class Vinardo(ScoringFunction):

	@ensure_software_installed("GNINA")
	def __init__(self, software_path: Path):
		super().__init__("Vinardo", "Vinardo", "min", (200, 20), software_path)
		self.software_path = software_path

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()

		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		temp_dir = self.create_temp_dir()
		try:
			split_files_folder = split_sdf_str(Path(temp_dir), sdf, n_cpus)
			split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

			global vinardo_rescoring_splitted

			def vinardo_rescoring_splitted(split_file, protein_file):
				results = Path(temp_dir) / f"{Path(split_file).stem}_{self.column_name}.sdf"
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

			parallel_executor(vinardo_rescoring_splitted,
					split_files_sdfs,
					n_cpus,
					display_name=self.column_name,
					protein_file=protein_file)

			try:
				vinardo_dataframes = [
					PandasTools.LoadSDF(str(Path(temp_dir) / file),
						idName="Pose ID",
						molColName=None,
						includeFingerprints=False,
						embedProps=False)
					for file in os.listdir(temp_dir)
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

			toc = time.perf_counter()
			printlog(f"Rescoring with Vinardo complete in {toc - tic:0.4f}!")
			return vinardo_rescoring_results
		finally:
			self.remove_temp_dir(temp_dir)


# Usage:
# vinardo = Vinardo()
# results = vinardo.rescore(sdf_file, n_cpus, software=software_path, protein_file=protein_file_path)
