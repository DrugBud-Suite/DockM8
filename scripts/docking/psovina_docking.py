import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
from meeko import PDBQTMolecule, RDKitMolCreate
from rdkit import RDLogger

from scripts.docking.docking_function import DockingFunction
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.file_splitting import split_pdbqt_str


class PsovinaDocking(DockingFunction):

	def __init__(self, software_path: Path):
		super().__init__("PSOVINA", software_path)

	def dock_batch(self,
		batch_file: Path,
		protein_file: Path,
		pocket_definition: Dict[str, list],
		exhaustiveness: int,
		n_poses: int) -> Path:

		temp_dir = self.create_temp_dir()
		results_folder = temp_dir / "docked"
		results_folder.mkdir(parents=True, exist_ok=True)

		# Convert molecules to pdbqt format
		try:
			pdbqt_files = convert_molecules(batch_file, temp_dir, "sdf", "pdbqt", self.software_path)
			if isinstance(pdbqt_files, Path):
				pdbqt_files = [pdbqt_files]
		except Exception as e:
			printlog(f"Failed to convert sdf file to .pdbqt: {e}")
			self.remove_temp_dir(temp_dir)
			return None

		protein_file_pdbqt = temp_dir / "protein.pdbqt"
		try:
			convert_molecules(protein_file, protein_file_pdbqt, "pdb", "pdbqt", self.software_path)
		except Exception as e:
			printlog(f"Failed to convert protein file to .pdbqt: {e}")
			self.remove_temp_dir(temp_dir)
			return None

		output_files = []
		for file in pdbqt_files:
			output_file = results_folder / f"{file.stem}_PSOVINA.pdbqt"
			psovina_cmd = (f"{self.software_path}/psovina"
							f" --receptor {protein_file_pdbqt}"
							f" --ligand {file}"
							f" --out {output_file}"
							f" --center_x {pocket_definition['center'][0]}"
							f" --center_y {pocket_definition['center'][1]}"
							f" --center_z {pocket_definition['center'][2]}"
							f" --size_x {pocket_definition['size'][0]}"
							f" --size_y {pocket_definition['size'][1]}"
							f" --size_z {pocket_definition['size'][2]}"
							f" --exhaustiveness {exhaustiveness}"
							" --cpu 1 --seed 1 --energy_range 10"
							f" --num_modes {n_poses}")
			try:
				subprocess.run(psovina_cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
				output_files.append(output_file)
			except subprocess.CalledProcessError as e:
				printlog(f"PSOVINA docking failed: {e}")
				self.remove_temp_dir(temp_dir)
				return None

		for files in output_files:
			split_pdbqt_str(files)

		return results_folder

	def process_docking_result(self, results_folder: Path, n_poses: int) -> pd.DataFrame:
		RDLogger.DisableLog("rdApp.*")
		try:
			psovina_poses = []
			for pose_file in results_folder.glob("*.pdbqt"):
				pdbqt_mol = PDBQTMolecule.from_file(pose_file, name=pose_file.stem, skip_typing=True)
				rdkit_mol = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
				with open(pose_file) as file:
					affinity = next(line.split()[3] for line in file if "REMARK VINA RESULT:" in line)

				# Extract ID from the filename (assuming filename format: {ID}_*.pdbqt)
				mol_id = pose_file.stem.split("_")[0]

				psovina_poses.append({
					"Pose ID": pose_file.stem,
					"Molecule": rdkit_mol[0],
					"PSOVINA_Affinity": float(affinity),
					"ID": mol_id})

			df = pd.DataFrame(psovina_poses)
			df = df.sort_values(['ID', 'PSOVINA_Affinity'])
			df['Pose Rank'] = df.groupby('ID').cumcount() + 1
			df = df[df['Pose Rank'] <= n_poses]
			df['Pose ID'] = df.apply(lambda row: f"{row['ID']}_PSOVINA_{row['Pose Rank']}", axis=1)

			return df[['Pose ID', 'ID', 'Molecule', 'PSOVINA_Affinity']]

		except Exception as e:
			printlog(f"ERROR: Failed to process PSOVINA result file {results_folder}: {str(e)}")
			return pd.DataFrame()
		finally:
			self.remove_temp_dir(results_folder.parent)
