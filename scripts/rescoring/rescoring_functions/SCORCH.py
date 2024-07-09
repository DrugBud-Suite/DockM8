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

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class SCORCH(ScoringFunction):

	def __init__(self):
		super().__init__("SCORCH", "SCORCH", "max", (0, 1))

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> pd.DataFrame:
		tic = time.perf_counter()
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		temp_dir = self.create_temp_dir()
		try:
			SCORCH_protein = Path(temp_dir) / "protein.pdbqt"
			convert_molecules(Path(str(protein_file)), SCORCH_protein, "pdb", "pdbqt")
			# Convert ligands to pdbqt
			split_files_folder = Path(temp_dir) / f"split_{Path(sdf).stem}"
			split_files_folder.mkdir(exist_ok=True)
			convert_molecules(sdf, split_files_folder, "sdf", "pdbqt")
			# Run SCORCH

			SCORCH_command = f"cd {software}/SCORCH-1.0.0/ && {sys.executable} ./scorch.py --receptor {SCORCH_protein} --ligand {split_files_folder} --out {temp_dir}/scoring_results.csv --threads {n_cpus} --return_pose_scores"
			subprocess.call(SCORCH_command, shell=True                        #, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
							)
			                                                                     # Clean data
			SCORCH_scores = pd.read_csv(Path(temp_dir) / "scoring_results.csv")
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
