import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import pandas as pd
from rdkit.Chem import PandasTools

# Assuming the same directory structure as in previous implementations
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.file_splitting import split_sdf_single_str
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.utilities import delete_files

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class CENsible(ScoringFunction):

	def __init__(self):
		super().__init__("CENsible", "CENsible", "max", (0, 20))
		self.censible_folder = self.check_and_download_censible()

	def check_and_download_censible(self):
		censible_folder = dockm8_path / "software" / "censible"
		if not os.path.exists(censible_folder):
			printlog("CENsible folder not found. Downloading...")
			download_url = "https://github.com/durrantlab/censible/archive/refs/heads/main.zip"
			download_path = dockm8_path / "software" / "censible.zip"
			urllib.request.urlretrieve(download_url, download_path)
			printlog("Download complete. Extracting...")
			with zipfile.ZipFile(download_path, 'r') as zip_ref:
				zip_ref.extractall(path=dockm8_path / "software")
			os.rename(dockm8_path / "software" / "censible-main", censible_folder)
			printlog("Extraction complete. Removing zip file...")
			os.remove(download_path)
			subprocess.run([sys.executable, "-m", "pip", "install", "-r", censible_folder / "requirements_predict.txt"])
			printlog("CENsible setup complete.")
		else:
			printlog("CENsible folder already exists.")
		return censible_folder

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		rescoring_folder = Path(kwargs.get("rescoring_folder"))
		protein_file = Path(kwargs.get("protein_file"))
		smina_path = self.find_executable("smina")
		obabel_path = self.find_executable("obabel")

		if smina_path is None or obabel_path is None:
			raise FileNotFoundError("smina or obabel executable not found. Please ensure they're installed.")

		(rescoring_folder / f"{self.column_name}_rescoring").mkdir(parents=True, exist_ok=True)
		split_files_folder = split_sdf_single_str((rescoring_folder / f"{self.column_name}_rescoring"), sdf)
		split_files_sdfs = [Path(split_files_folder) / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]
		global censible_rescoring_splitted
		def censible_rescoring_splitted(split_file, protein_file):
			df = PandasTools.LoadSDF(str(split_file), idName="Pose ID", molColName=None)
			df = df[["Pose ID"]]

			with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as temp_pdb:
				convert_molecules(split_file, Path(temp_pdb.name), "sdf", "pdb")
				with tempfile.TemporaryDirectory() as temp_dir:
					temp_protein = Path(temp_dir) / protein_file.name
					shutil.copy(protein_file, temp_protein)
					censible_command = [
						sys.executable,
						self.censible_folder / "predict.py",
						"--ligpath",
						temp_pdb.name,
						"--recpath",
						str(temp_protein),
						"--smina_exec_path",
						smina_path,
						"--obabel_exec_path",
						obabel_path,
						"--use_cpu"]
					process = subprocess.Popen(censible_command,
							stdout=subprocess.PIPE,
							stderr=subprocess.PIPE,
							text=True)
					stdout, stderr = process.communicate()
					score = None
					for line in stdout.split('\n'):
						if "score:" in line:
							match = re.search(r'score:\s+(-?\d+\.?\d*)', line)
							if match:
								score = float(match.group(1))
								break
					if score is None:
						printlog(f"Warning: Could not parse score for file {split_file}. Setting to None.")
			os.unlink(temp_pdb.name)
			df[self.column_name] = [score]
			output_csv = str(rescoring_folder / f"{self.column_name}_rescoring" / (str(split_file.stem) + "_score.csv"))
			df.to_csv(output_csv, index=False)

		parallel_executor(censible_rescoring_splitted, split_files_sdfs, n_cpus, protein_file=protein_file)

		scores_folder = rescoring_folder / f"{self.column_name}_rescoring"
		score_files = list(scores_folder.glob("*_score.csv"))
		if not score_files:
			printlog("No CSV files found with names ending in '_score.csv' in the specified folder.")
			return pd.DataFrame()
		combined_scores_df = pd.concat([pd.read_csv(file) for file in score_files], ignore_index=True)
		CENsible_rescoring_results = scores_folder / f"{self.column_name}_scores.csv"
		combined_scores_df.to_csv(CENsible_rescoring_results, index=False)

		delete_files(rescoring_folder / f"{self.column_name}_rescoring", f"{self.column_name}_scores.csv")
		toc = time.perf_counter()
		printlog(f"Rescoring with CENsible complete in {toc-tic:0.4f}!")
		return combined_scores_df

	@staticmethod
	def find_executable(name):
		try:
			result = subprocess.run(['which', name], capture_output=True, text=True, check=True)
			return result.stdout.strip()
		except subprocess.CalledProcessError:
			try:
				result = subprocess.run(['where', name], capture_output=True, text=True, check=True)
				return result.stdout.strip().split('\n')[0]
			except subprocess.CalledProcessError:
				return None
