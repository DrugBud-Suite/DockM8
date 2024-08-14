import subprocess
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.docking.docking_function import DockingFunction
from scripts.setup.software_manager import ensure_software_installed
from scripts.utilities.logging import printlog
from scripts.utilities.utilities import parallel_SDF_loader


class GninaDocking(DockingFunction):

	def __init__(self, software_path: Path):
		super().__init__("GNINA", software_path)
		self.software_path = software_path
		ensure_software_installed("GNINA", software_path)

	def dock_batch(self,
		batch_file: Path,
		protein_file: Path,
		pocket_definition: Dict[str, list],
		exhaustiveness: int,
		n_poses: int) -> Path:

		RDLogger.DisableLog("rdApp.*")
		temp_dir = self.create_temp_dir()
		results_path = temp_dir / f"{batch_file.stem}_gnina.sdf"

		gnina_cmd = (f"{self.software_path}/gnina"
			f" --receptor {protein_file}"
			f" --ligand {batch_file}"
			f" --out {results_path}"
			f" --center_x {pocket_definition['center'][0]}"
			f" --center_y {pocket_definition['center'][1]}"
			f" --center_z {pocket_definition['center'][2]}"
			f" --size_x {pocket_definition['size'][0]}"
			f" --size_y {pocket_definition['size'][1]}"
			f" --size_z {pocket_definition['size'][2]}"
			f" --exhaustiveness {exhaustiveness}"
			" --cpu 1 --seed 1"
			f" --num_modes {n_poses}"
			" --cnn_scoring rescore --cnn crossdock_default2018 --no_gpu")

		try:
			subprocess.call(gnina_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		except Exception as e:
			printlog(f"GNINA docking failed: {e}")
			self.remove_temp_dir(temp_dir)
			return None

		return results_path

	def process_docking_result(self, result_file: Path, n_poses: int) -> pd.DataFrame:
		RDLogger.DisableLog("rdApp.*")
		try:
			df = parallel_SDF_loader(result_file, molColName="Molecule", idName="ID")
			df['CNN-Score'] = df['CNNscore'].astype(float)

			# Sort by CNN-Score (higher is better) and rank within each ID group
			df = df.sort_values(['ID', 'CNN-Score'], ascending=[True, False])
			df['Pose Rank'] = df.groupby('ID').cumcount() + 1
			df = df[df['Pose Rank'] <= n_poses]

			df['Pose ID'] = df.apply(lambda row: f"{row['ID']}_GNINA_{row['Pose Rank']}", axis=1)

			return df[['Pose ID', 'ID', 'Molecule', 'CNN-Score']]
		except Exception as e:
			printlog(f"ERROR: Failed to process GNINA result file {result_file}: {str(e)}")
			return pd.DataFrame()
