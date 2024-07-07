import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
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
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.utilities import delete_files

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ITScoreAff(ScoringFunction):

	def __init__(self):
		super().__init__("ITScoreAff", "ITScoreAff", "min", (-200, 100))
		self.itscore_folder = self.check_and_download_itscoreAff()

	def check_and_download_itscoreAff(self):
		itscore_folder = dockm8_path / "software" / "ITScoreAff_v1.0"

		if not os.path.exists(itscore_folder):
			printlog("ITScoreAff_v1.0 folder not found. Downloading...")
			download_url = "http://huanglab.phys.hust.edu.cn/ITScoreAff/ITScoreAff_v1.0.tar.gz"
			download_path = dockm8_path / "software" / "ITScoreAff_v1.0.tar.gz"
			urllib.request.urlretrieve(download_url, download_path)
			printlog("Download complete. Extracting...")

			with tarfile.open(download_path, "r:gz") as tar:
				tar.extractall(path=dockm8_path / "software")

			printlog("Extraction complete. Removing tarball...")
			os.remove(download_path)

			executable_path = itscore_folder / "ITScoreAff"
			os.chmod(executable_path, os.stat(executable_path).st_mode | stat.S_IEXEC)
			printlog(f"Changed permissions for {executable_path}")

			printlog("ITScoreAff_v1.0 setup complete.")
		else:
			pass

		return itscore_folder

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		rescoring_folder = Path(kwargs.get("rescoring_folder"))
		protein_file = Path(kwargs.get("protein_file"))

		(rescoring_folder / f"{self.column_name}_rescoring").mkdir(parents=True, exist_ok=True)
		split_files_folder = split_sdf_str((rescoring_folder / f"{self.column_name}_rescoring"), sdf, n_cpus)
		split_files_sdfs = [Path(split_files_folder) / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

		with tempfile.TemporaryDirectory(prefix="itscoreAff_") as temp_dir:
			protein_mol2 = Path(temp_dir) / 'protein.mol2'
			convert_molecules(protein_file, protein_mol2, protein_file.suffix[1:], 'mol2')
			global ITScoreAff_rescoring_splitted
			def ITScoreAff_rescoring_splitted(split_file, protein_mol2):
				df = PandasTools.LoadSDF(str(split_file), idName="Pose ID", molColName=None)
				df = df[["Pose ID"]]

				ligand_mol2 = Path(temp_dir) / f'{split_file.stem}.mol2'
				convert_molecules(split_file, ligand_mol2, 'sdf', 'mol2')

				itscoreAff_command = f"{self.itscore_folder}/ITScoreAff {protein_mol2} {ligand_mol2}"
				process = subprocess.Popen(itscoreAff_command,
					stdout=subprocess.PIPE,
					stderr=subprocess.PIPE,
					shell=True)
				stdout, stderr = process.communicate()

				scores = []
				output = stdout.decode().splitlines()
				for line in output[1:]:                                                                             # Skip the header line
					parts = line.split()
					if len(parts) >= 3:
						try:
							score = round(float(parts[2]), 2)
							scores.append(score)
						except ValueError:
							printlog(
								f"Warning: Could not convert '{parts[2]}' to float for file {split_file}. Skipping this score."
							)
							scores.append(None)
				df[self.column_name] = scores
				output_csv = str(rescoring_folder / f"{self.column_name}_rescoring" /
					(str(split_file.stem) + "_scores.csv"))
				df.to_csv(output_csv, index=False)

			parallel_executor(ITScoreAff_rescoring_splitted, split_files_sdfs, n_cpus, protein_mol2=protein_mol2)

		scores_folder = rescoring_folder / f"{self.column_name}_rescoring"
		score_files = list(scores_folder.glob("*_scores.csv"))
		if not score_files:
			printlog("No CSV files found with names ending in '_scores.csv' in the specified folder.")
			return pd.DataFrame()

		combined_scores_df = pd.concat([pd.read_csv(file) for file in score_files], ignore_index=True)
		ITScoreAff_rescoring_results = scores_folder / f"{self.column_name}_scores.csv"
		combined_scores_df.to_csv(ITScoreAff_rescoring_results, index=False)

		delete_files(rescoring_folder / f"{self.column_name}_rescoring", f"{self.column_name}_scores.csv")

		toc = time.perf_counter()
		printlog(f"Rescoring with ITScoreAff complete in {toc-tic:0.4f}!")
		return combined_scores_df
