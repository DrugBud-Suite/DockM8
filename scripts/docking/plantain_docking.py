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
from scripts.pocket_finding.utils import extract_pocket
from scripts.utilities.utilities import parallel_SDF_loader

class PlantainDocking(DockingFunction):

	def __init__(self, software_path: Path):
		super().__init__("PLANTAIN", software_path)
		self.software_path = software_path
		ensure_software_installed("PLANTAIN", software_path)

	def dock_batch(self,
					batch_file: Path,
					protein_file: Path,
					pocket_definition: Dict[str, list],
					exhaustiveness: int,
					n_poses: int) -> Path:
		RDLogger.DisableLog("rdApp.*")
		temp_dir = self.create_temp_dir()

		# Convert SDF to SMILES
		smiles_file = temp_dir / f"{batch_file.stem}_compounds.smi"
		mols = Chem.SDMolSupplier(str(batch_file))
		with open(smiles_file, 'w') as f:
			for mol in mols:
				if mol is not None:
					smi = Chem.MolToSmiles(mol)
					name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
					f.write(f"{smi} {name}\n")

		# Prepare the pocket definition
		protein_pocket = str(protein_file).replace(".pdb", "_pocket.pdb")

		if not os.path.isfile(protein_pocket):
			protein_pocket = extract_pocket(protein_file, pocket_definition, protein_pocket)

		# Construct the PLANTAIN command
		predictions_dir = temp_dir / f"{batch_file.stem}_predictions"
		plantain_cmd = (f"cd {self.software_path}/plantain && python ./inference.py "
						f"{smiles_file} {protein_pocket} "
						f"--out {predictions_dir} "
						f"--num_workers 1 --no_gpu")

		try:
			# Execute the PLANTAIN command
			subprocess.call(plantain_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

			# Combine all output SDF files into one
			output_file = temp_dir / f"{batch_file.stem}_poses.sdf"
			writer = Chem.SDWriter(str(output_file))
			for sdf_file in predictions_dir.glob("*.sdf"):
				for mol in Chem.SDMolSupplier(str(sdf_file)):
					if mol is not None:
						writer.write(mol)
			writer.close()
			return output_file
		except Exception as e:
			printlog(f"PLANTAIN docking failed: {e}")
			self.remove_temp_dir(temp_dir)
			return None

	def process_docking_result(self, result_file: Path, n_poses: int) -> pd.DataFrame:
		try:
			df = parallel_SDF_loader(result_file,
										molColName="Molecule",
										smilesName="SMILES",
										idName='ID',
										strictParsing=False)
			# Add ranking based on the order of poses in the file
			df['PLANTAIN_Rank'] = df.groupby('ID').cumcount() + 1

			# Keep only the top n_poses for each molecule
			df = df[df['PLANTAIN_Rank'] <= n_poses]

			# Create Pose ID
			df['Pose ID'] = df.apply(lambda row: f"{row['ID']}PLANTAIN{int(row['PLANTAIN_Rank'])}", axis=1)

			return df

		except Exception as e:
			printlog(f"ERROR: Failed to process PLANTAIN result file {result_file}: {str(e)}")
			return pd.DataFrame()     # Return an empty DataFrame on error
		finally:
			self.remove_temp_dir(result_file.parent)
