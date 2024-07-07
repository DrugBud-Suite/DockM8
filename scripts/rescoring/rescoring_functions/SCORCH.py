import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
from pandas import DataFrame

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.scoring_function import ScoringFunction
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.utilities import delete_files

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class SCORCH(ScoringFunction):

	def __init__(self):
		super().__init__("SCORCH", "SCORCH", "max", (0, 1))

	def rescore(self, sdf: str, n_cpus: int, **kwargs) -> DataFrame:
		tic = time.perf_counter()
		rescoring_folder = kwargs.get("rescoring_folder")
		software = kwargs.get("software")
		protein_file = kwargs.get("protein_file")

		SCORCH_rescoring_folder = rescoring_folder / f"{self.column_name}_rescoring"
		SCORCH_rescoring_folder.mkdir(parents=True, exist_ok=True)
		SCORCH_protein = SCORCH_rescoring_folder / "protein.pdbqt"
		convert_molecules(str(protein_file).replace(".pdb", "_pocket.pdb"), SCORCH_protein, "pdb", "pdbqt")
		# Convert ligands to pdbqt
		split_files_folder = SCORCH_rescoring_folder / f"split_{Path(sdf).stem}"
		split_files_folder.mkdir(exist_ok=True)
		convert_molecules(sdf, split_files_folder, "sdf", "pdbqt")
		# Run SCORCH

		SCORCH_command = f"cd {software}/SCORCH-1.0.0/ && {sys.executable} ./scorch.py --receptor {SCORCH_protein} --ligand {split_files_folder} --out {SCORCH_rescoring_folder}/scoring_results.csv --threads {n_cpus} --return_pose_scores"
		subprocess.call(SCORCH_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		# Clean data
		SCORCH_scores = pd.read_csv(SCORCH_rescoring_folder / "scoring_results.csv")
		SCORCH_scores = SCORCH_scores.rename(columns={"Ligand_ID": "Pose ID", "SCORCH_pose_score": self.column_name})
		SCORCH_scores = SCORCH_scores[[self.column_name, "Pose ID"]]
		SCORCH_rescoring_results = SCORCH_rescoring_folder / f"{self.column_name}_scores.csv"
		SCORCH_scores.to_csv(SCORCH_rescoring_results, index=False)
		delete_files(SCORCH_rescoring_folder, f"{self.column_name}_scores.csv")
		toc = time.perf_counter()
		printlog(f"Rescoring with SCORCH complete in {toc-tic:0.4f}!")
		return SCORCH_scores
