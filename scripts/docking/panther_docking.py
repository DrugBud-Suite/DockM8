import subprocess
from pathlib import Path
from typing import Dict

import pandas as pd
from rdkit import RDLogger
from rdkit.Chem import PandasTools

from scripts.docking.docking_function import DockingFunction
from scripts.utilities.logging import printlog


class PantherDocking(DockingFunction):

	def __init__(self, software_path: Path):
		super().__init__("PANTHER", software_path)
		self.negative_image = None

	def generate_negative_image(self, protein_file: Path, pocket_definition: Dict[str, list]) -> Path:
		temp_dir = self.create_temp_dir()

		# Read default PANTHER input file and modify it
		default_in = self.software_path / "default.in"
		with open(default_in, 'r') as f:
			lines = f.readlines()
		for i, line in enumerate(lines):
			if line.startswith("1-Pdb file"):
				lines[i] = f"1-Pdb file (-pfil):: {protein_file}\n"
			elif line.startswith("2-Radius library"):
				lines[i] = f"2-Radius library (-rlib):: {self.software_path}/panther/rad.lib\n"
			elif line.startswith("3-Angle library"):
				lines[i] = f"3-Angle library (-alib):: {self.software_path}/panther/angles.lib\n"
			elif line.startswith("4-Charge library file"):
				lines[i] = f"4-Charge library file (-chlib):: {self.software_path}/panther/charges.lib\n"
			elif line.startswith("5-Center(s)"):
				center = pocket_definition['center']
				lines[i] = f"5-Center(s) (-cent):: {center[0]} {center[1]} {center[2]}\n"
			elif line.startswith("9-Box radius"):
				box_size = pocket_definition['size'][0] // 2
				lines[i] = f"9-Box radius (-brad):: {box_size}\n"

		# Write the modified input file
		panther_input = temp_dir / "panther_input.in"
		with open(panther_input, 'w') as f:
			f.writelines(lines)

		# Run PANTHER
		negative_image = temp_dir / "negative_image.mol2"
		panther_cmd = f"conda run -n panther python {self.software_path}/panther/panther.py {panther_input} {negative_image}"

		try:
			process = subprocess.Popen(panther_cmd,
										shell=True,
										stdout=subprocess.PIPE,
										stderr=subprocess.PIPE,
										universal_newlines=True)
			stdout, stderr = process.communicate()

			# Extract mol2 data from the output
			mol2_start = stdout.find("@<TRIPOS>MOLECULE")
			mol2_end = stdout.rfind("INFO:")
			if mol2_start != -1 and mol2_end != -1:
				mol2_data = stdout[mol2_start:mol2_end].strip()
				with open(negative_image, 'w') as f:
					f.write(mol2_data)
				return negative_image
			else:
				raise RuntimeError("PANTHER failed to generate mol2 file")
		except Exception as e:
			printlog(f"PANTHER Negative Image Generation failed: {e}")
			self.remove_temp_dir(temp_dir)
			return None

	def dock_batch(self,
					batch_file: Path,
					protein_file: Path,
					pocket_definition: Dict[str, list],
					exhaustiveness: int,
					n_poses: int) -> Path:

		temp_dir = self.create_temp_dir()

		# Prepare input files
		ligs_decs_1_conf_prepped = temp_dir / f"{batch_file}-prepped.mol2"
		obabel_charge_cmd = f"obabel -isdf {batch_file} -O {ligs_decs_1_conf_prepped} --partialcharge mmff94"
		conformers_file = temp_dir / f"{batch_file}-prepped.mol2"
		obabel_conformers_cmd = f"obabel -imol2 {ligs_decs_1_conf_prepped} -omol2 -O {conformers_file} --confab --conf 100000 --rcutoff 1"

		try:
			subprocess.call(obabel_charge_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			subprocess.call(obabel_conformers_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		except Exception as e:
			printlog(f"PANTHER conformer generation failed: {e}")
			self.remove_temp_dir(temp_dir)
			return None

		# Generate negative image if it doesn't exist
		if self.negative_image is None:
			self.negative_image = self.generate_negative_image(protein_file, pocket_definition)
			if self.negative_image is None:
				return None

		# Run SHAEP
		shaep_executable = self.software_path / "shaep"
		if not shaep_executable.is_file():
			printlog("SHAEP executable not found. Please download SHAEP and ensure it is in DockM8's software folder.")
			self.remove_temp_dir(temp_dir)
			return None

		shaep_output_sdf = temp_dir / f"{batch_file}_shaep_results.sdf"
		shaep_output_txt = temp_dir / f"{batch_file}_shaep_results.txt"
		shaep_cmd = f"{shaep_executable} -q {self.negative_image} {conformers_file} -s {shaep_output_sdf} --nStructures {n_poses} --output-file {shaep_output_txt}"

		try:
			subprocess.call(shaep_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		except Exception as e:
			printlog(f"SHAEP similarity calculation to negative image failed: {e}")
			self.remove_temp_dir(temp_dir)
			return None

		return shaep_output_sdf

	def process_docking_result(self, result_file: Path, n_poses: int) -> pd.DataFrame:
		RDLogger.DisableLog('rdApp.*')
		try:
			df = PandasTools.LoadSDF(str(result_file), molColName="Molecule", smilesName="SMILES", strictParsing=False)
			# Sort by Similarity_best in descending order and create new rank
			df = df.sort_values('Similarity_best', ascending=False)
			df['PANTHER_Rank'] = df.groupby('ID')['Similarity_best'].rank(method='first', ascending=False)

			# Keep only the top n_poses for each molecule
			df = df[df['PANTHER_Rank'] <= n_poses]

			# Create Pose ID based on the new rank
			df['Pose ID'] = df.apply(lambda row: f"{row['ID']}_PANTHER_{int(row['PANTHER_Rank'])}", axis=1)

			# Rename Similarity_best to PANTHER_Score
			df = df.rename(columns={'Similarity_best': 'PANTHER_Score'})

			# Keep only necessary columns
			df = df[['Molecule', 'ID', 'Pose ID', 'PANTHER_Score']]

			return df

		except Exception as e:
			printlog(f"ERROR: Failed to process PANTHER result file {result_file}: {str(e)}")
			return pd.DataFrame()     # Return an empty DataFrame on error
		finally:
			self.remove_temp_dir(result_file.parent)
