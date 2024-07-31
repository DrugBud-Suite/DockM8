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
from scripts.utilities.molecule_conversion import convert_molecules

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class KORPL(ScoringFunction):

	@ensure_software_installed("KORP_PL")
	def __init__(self, software_path: Path):
		super().__init__("KORP-PL", "KORP-PL", "min", (200, -1000), software_path)
		self.software_path = software_path

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		protein_file = kwargs.get("protein_file")

		temp_dir = self.create_temp_dir()
		try:
			split_files_folder = split_sdf_str(Path(temp_dir), sdf, n_cpus)
			split_files_sdfs = [
				Path(split_files_folder) / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

			global KORPL_rescoring_splitted

			def KORPL_rescoring_splitted(split_file, protein_file):
				df = PandasTools.LoadSDF(str(split_file), idName="Pose ID", molColName=None)
				df = df[["Pose ID"]]
				mol2_file = convert_molecules(split_file,
												split_file.with_suffix("mol2"),
												"sdf",
												"mol2",
												self.software_path)
				korpl_command = (f"{self.software_path}/KORP-PL --receptor {protein_file} --ligand {mol2_file}")
				print(korpl_command)
				process = subprocess.Popen(korpl_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
				stdout, stderr = process.communicate()
				energies = []
				output = stdout.decode().splitlines()
				for line in output:
					if line.startswith("model"):
						parts = line.split(",")
						energy = round(float(parts[1].split("=")[1]), 2)
						energies.append(energy)
				df[self.column_name] = energies
				output_csv = str(Path(temp_dir) / (str(split_file.stem) + "_scores.csv"))
				df.to_csv(output_csv, index=False)

			parallel_executor(KORPL_rescoring_splitted,
								split_files_sdfs,
								n_cpus,
								display_name=self.column_name,
								protein_file=protein_file)

			print("Combining KORPL scores")
			score_files = list(Path(temp_dir).glob("*_scores.csv"))
			if not score_files:
				print("No CSV files found with names ending in '_scores.csv' in the specified folder.")
				return pd.DataFrame()
			else:
				combined_scores_df = pd.concat([pd.read_csv(file) for file in score_files], ignore_index=True)

			toc = time.perf_counter()
			printlog(f"Rescoring with KORPL complete in {toc-tic:0.4f}!")
			return combined_scores_df
		finally:
			self.remove_temp_dir(temp_dir)


# Usage:
# korpl = KORPL()
# results = korpl.rescore(sdf_file, n_cpus, protein_file=protein_file_path)
