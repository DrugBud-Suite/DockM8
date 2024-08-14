import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.docking.docking_function import DockingFunction
from scripts.setup.software_manager import ensure_software_installed
from scripts.utilities.logging import printlog
from scripts.utilities.utilities import parallel_SDF_loader


class FABindDocking(DockingFunction):

	def __init__(self, software_path: Path):
		super().__init__("FABind", software_path)
		self.software_path = software_path
		ensure_software_installed("FABind", software_path)

	def dock_batch(self,
		batch_file: Path,
		protein_file: Path,
		pocket_definition: Dict[str, list],
		exhaustiveness: int,
		n_poses: int) -> Path:

		temp_dir = self.create_temp_dir()

		# Create necessary folders
		pdb_dir = temp_dir / "pdb"
		pdb_dir.mkdir(parents=True, exist_ok=True)
		save_mols_dir = temp_dir / "mol"
		save_mols_dir.mkdir(parents=True, exist_ok=True)

		# Copy protein file
		protein_file_path = Path(protein_file)
		pdb_file = pdb_dir / protein_file_path.name
		pdb_file.write_bytes(protein_file_path.read_bytes())

		# Prepare input files
		index_csv = temp_dir / f"{batch_file.stem}_index.csv"

		# Create index CSV from input SDF
		df = parallel_SDF_loader(batch_file, molColName='Molecule')
		df['Cleaned_SMILES'] = df['Molecule'].apply(
			lambda mol: Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol))))
		df['pdb_id'] = protein_file_path.stem
		csv_df = df[['Cleaned_SMILES', 'pdb_id', 'ID']]
		csv_df.columns = ['SMILES', 'pdb_id', 'ligand_id']
		csv_df.to_csv(index_csv, index=False)

		# Define commands
		conda_activate_cmd = "conda run -n fabind "
		preprocess_mol_cmd = (
			f"{conda_activate_cmd} python {self.software_path}/FABind/FABind_plus/fabind/inference_preprocess_mol_confs.py "
			f"--index_csv {index_csv} "
			f"--save_mols_dir {save_mols_dir} "
			f"--num_threads {1}")
		preprocess_protein_cmd = (
			f"{conda_activate_cmd} python {self.software_path}/FABind/FABind_plus/fabind/inference_preprocess_protein.py "
			f"--pdb_file_dir {pdb_dir} "
			f"--save_pt_dir {temp_dir}")
		inference_cmd = (
			f"{conda_activate_cmd} python {self.software_path}/FABind/FABind_plus/fabind/inference_fabind.py "
			f"--ckpt {self.software_path}/FABind/FABind_plus/ckpt/fabind_plus_best_ckpt.bin "
			f"--batch_size 8 "
			f"--post-optim "
			f"--write-mol-to-file "
			f"--sdf-output-path-post-optim {temp_dir} "
			f"--index-csv {index_csv} "
			f"--preprocess-dir {temp_dir} ")

		# Execute commands
		try:
			subprocess.run(preprocess_mol_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			subprocess.run(preprocess_protein_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
			subprocess.run(inference_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
		except Exception as e:
			printlog(f"FABind docking failed: {e}")
			self.remove_temp_dir(temp_dir)
			return None

		return temp_dir

	def process_docking_result(self, result_file: Path, n_poses: int) -> pd.DataFrame:
		RDLogger.DisableLog("rdApp.*")
		try:
			fabind_poses = pd.DataFrame()
			for file in result_file.glob("*.sdf"):
				try:
					df = parallel_SDF_loader(file, molColName="Molecule", smilesName="SMILES", strictParsing=False)

					ligand_id = file.stem
					df["ID"] = ligand_id
					df["Pose ID"] = f"{ligand_id}_FABind_1"
					fabind_poses = pd.concat([fabind_poses, df])
				except Exception as e:
					printlog(
						f"ERROR: Failed to process FABind result file {file} due to Exception: {e}, moving to next file"
					)
					pass

			return fabind_poses

		except Exception as e:
			printlog(f"ERROR: Failed to process FABind result files in {result_file}: {str(e)}")
			return pd.DataFrame()     # Return an empty DataFrame on error
		finally:
			self.remove_temp_dir(result_file)
