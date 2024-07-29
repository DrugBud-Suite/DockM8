import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.setup.software_manager import ensure_software_installed

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class SCORCH(ScoringFunction):

	@ensure_software_installed("SCORCH")
	def __init__(self, software_path: Path):
		super().__init__("SCORCH", "SCORCH", "max", (0, 1), software_path)
		self.software_path = software_path

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		protein_file = kwargs.get("protein_file")

		temp_dir = self.create_temp_dir()
		try:
			SCORCH_protein = Path(temp_dir) / "protein.pdbqt"
			try:
				convert_molecules(Path(str(protein_file)), SCORCH_protein, "pdb", "pdbqt", self.software_path)
			except Exception as e:
				printlog(f"Error converting protein file to .pdbqt: {str(e)}")
				return pd.DataFrame()

			# Convert ligands to pdbqt
			split_files_folder = Path(temp_dir) / f"split_{Path(sdf).stem}"
			split_files_folder.mkdir(exist_ok=True)
			try:
				convert_molecules(sdf, split_files_folder, "sdf", "pdbqt", self.software_path)
			except Exception as e:
				printlog(f"Error converting ligand file to .pdbqt: {str(e)}")
				return pd.DataFrame()

			# Run SCORCH
			SCORCH_command = f"cd {self.software_path}/SCORCH-1.0.0/ && {sys.executable} ./scorch.py --receptor {SCORCH_protein} --ligand {split_files_folder} --out {temp_dir}/scoring_results.csv --threads {n_cpus} --return_pose_scores"
			try:
				subprocess.run(SCORCH_command,
					shell=True,
					check=True,
					stdout=subprocess.DEVNULL,
					stderr=subprocess.STDOUT)
			except subprocess.CalledProcessError as e:
				printlog(f"Error running SCORCH command: {str(e)}")
				return pd.DataFrame()

			# Clean data
			results_file = Path(temp_dir) / "scoring_results.csv"
			if not results_file.exists():
				printlog(f"Results file not found: {results_file}")
				return pd.DataFrame()

			SCORCH_scores = pd.read_csv(results_file)
			SCORCH_scores = SCORCH_scores.rename(columns={
				"Ligand_ID": "Pose ID", "SCORCH_pose_score": self.column_name})
			SCORCH_scores = SCORCH_scores[[self.column_name, "Pose ID"]]

			toc = time.perf_counter()
			printlog(f"Rescoring with SCORCH complete in {toc-tic:0.4f}!")
			return SCORCH_scores
		finally:
			self.remove_temp_dir(temp_dir)


# Usage:
# scorch = SCORCH()
# results = scorch.rescore(sdf_file, n_cpus, software=software_path, protein_file=protein_file_path)
