import subprocess
from pathlib import Path
from typing import Dict
import sys
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import RDLogger

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.docking.docking_function import DockingFunction
from scripts.utilities.logging import printlog
from scripts.setup.software_manager import ensure_software_installed
from scripts.utilities.utilities import parallel_SDF_loader

class SminaDocking(DockingFunction):

	@ensure_software_installed("SMINA")
	def __init__(self, software_path: Path):
		super().__init__("SMINA", software_path)

	def dock_batch(self,
					batch_file: Path,
					protein_file: Path,
					pocket_definition: Dict[str, list],
					exhaustiveness: int,
					n_poses: int) -> Path:
		RDLogger.DisableLog("rdApp.*")
		temp_dir = self.create_temp_dir()
		results_path = temp_dir / f"{batch_file.stem}_smina.sdf"

		smina_cmd = (f"{self.software_path}/gnina"
						f" --receptor {protein_file}"
						f" --ligand {batch_file}"
						f" --out {results_path}"
						f' --center_x {pocket_definition["center"][0]}'
						f' --center_y {pocket_definition["center"][1]}'
						f' --center_z {pocket_definition["center"][2]}'
						f' --size_x {pocket_definition["size"][0]}'
						f' --size_y {pocket_definition["size"][1]}'
						f' --size_z {pocket_definition["size"][2]}'
						f" --exhaustiveness {exhaustiveness}"
						" --cpu 1 --seed 1"
						f" --num_modes {n_poses}"
						" --cnn_scoring none --no_gpu")

		try:
			subprocess.call(smina_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		except Exception as e:
			printlog(f"SMINA docking failed for batch {batch_file}: {e}")
			self.remove_temp_dir(temp_dir)
			return None

		return results_path

	def process_docking_result(self, result_file: Path, n_poses: int) -> pd.DataFrame:
		RDLogger.DisableLog("rdApp.*")
		try:
			df = parallel_SDF_loader(result_file, molColName="Molecule", idName="ID")
			df['SMINA_Affinity'] = df['minimizedAffinity'].astype(float)

			# Sort by SMINA_Affinity (lower is better) and rank within each ID group
			df = df.sort_values(['ID', 'SMINA_Affinity'])
			df['Pose Rank'] = df.groupby('ID').cumcount() + 1
			df = df[df['Pose Rank'] <= n_poses]

			df['Pose ID'] = df.apply(lambda row: f"{row['ID']}_SMINA_{row['Pose Rank']}", axis=1)

			return df[['Pose ID', 'ID', 'Molecule', 'SMINA_Affinity']]
		except Exception as e:
			printlog(f"ERROR: Failed to process SMINA result file {result_file}: {str(e)}")
			return pd.DataFrame()
