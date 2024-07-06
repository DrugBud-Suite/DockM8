import subprocess
import sys
from pathlib import Path
from typing import Union

import pandas as pd
from rdkit.Chem import PandasTools

# Assuming similar path structure as in the provided code
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog
from scripts.utilities.utilities import delete_files


def panther_docking(split_file: Path,
					w_dir: Path,
					protein_file: str,
					pocket_coordinates: dict,
					software: Path,
					n_poses: int):
	"""
	Performs docking using the PANTHER software.

	Args:
		split_file (Path): The path to the split file.
		w_dir (Path): The working directory.
		protein_file (str): The path to the protein file.
		pocket_coordinates (dict): A dictionary containing the center and size of the pocket.
		software (Path): The path to the PANTHER software.
		n_poses (int): The number of poses to generate.

	Returns:
		None
	"""
	# Create necessary folders
	panther_folder = w_dir / "panther"
	panther_folder.mkdir(parents=True, exist_ok=True)
	# Create a unique temporary folder based on the split file name
	split_name = split_file.stem if split_file else "final_library"
	temp_files_dir = panther_folder / f"temp_files_{split_name}"
	temp_files_dir.mkdir(parents=True, exist_ok=True)
	# Prepare input files
	input_file = split_file if split_file else w_dir / "final_library.sdf"

	try:
		# Command #3: Add partial charges for the ligands
		ligs_decs_1_conf_prepped = temp_files_dir / "ligs-decs-1-conf-prepped.mol2"
		obabel_charge_cmd = f"obabel -isdf {input_file} -O {ligs_decs_1_conf_prepped} --partialcharge mmff94"
		subprocess.call(obabel_charge_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		# Command #4: Generate ligand 3D conformers
		conformers_file = temp_files_dir / "ligs-decs-multi-prepped.mol2"
		obabel_conformers_cmd = f"obabel -imol2 {ligs_decs_1_conf_prepped} -omol2 -O {conformers_file} --confab --conf 100000 --rcutoff 1"
		subprocess.call(obabel_conformers_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	except Exception as e:
		printlog(f"PANTHER conformer generation failed: {e}")

	try:
		# Read default PANTHER input file
		default_in = software / "default.in"
		# Modify the default.in file with correct protein path and center coordinates
		with open(default_in, 'r') as f:
			lines = f.readlines()
		for i, line in enumerate(lines):
			if line.startswith("1-Pdb file"):
				lines[i] = f"1-Pdb file (-pfil):: {protein_file}\n"
			elif line.startswith("2-Radius library"):
				lines[i] = f"2-Radius library (-rlib):: {software}/panther/rad.lib\n"
			elif line.startswith("3-Angle library"):
				lines[i] = f"3-Angle library (-alib):: {software}/panther/angles.lib\n"
			elif line.startswith("4-Charge library file"):
				lines[i] = f"4-Charge library file (-chlib):: {software}/panther/charges.lib\n"
			elif line.startswith("5-Center(s)"):
				center = pocket_coordinates['center']
				lines[i] = f"5-Center(s) (-cent):: {center[0]} {center[1]} {center[2]}\n"
			elif line.startswith("9-Box radius"):
				box_size = pocket_coordinates['size'][0] // 2
				lines[i] = f"9-Box radius (-brad):: {box_size}\n"
		# Write the modified input file
		panther_input = temp_files_dir / "panther_input.in"
		with open(panther_input, 'w') as f:
			f.writelines(lines)
		# Run PANTHER and capture its output
		negative_image = panther_folder / f"negative_image_{split_name}.mol2"
		panther_cmd = f"conda run -n panther python {software}/panther/panther.py {panther_input} {negative_image}"
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
			# Write the extracted mol2 data to the file
			with open(negative_image, 'w') as f:
				f.write(mol2_data)
			printlog(f"Negative image written to {negative_image}")
		else:
			printlog(f"PANTHER stdout: {stdout}")
			printlog(f"PANTHER stderr: {stderr}")
			raise RuntimeError("PANTHER failed to generate mol2 file")
	except Exception as e:
		printlog(f"PANTHER Negative Image Generation failed: {e}")

	try:

		# Check if SHAEP executable exists
		shaep_executable = software / "shaep"
		if not shaep_executable.is_file():
			printlog(
				"SHAEP executable not found. Please download SHAEP and ensure it is in DockM8's software folder. SHAEP can be found here: https://users.abo.fi/mivainio/shaep/download.php"
			)
		else:
			# Run SHAEP
			shaep_output_sdf = panther_folder / f"shaep_results_{split_name}.sdf"
			shaep_output_txt = panther_folder / f"shaep_results_{split_name}.txt"
			shaep_cmd = f"{software}/shaep -q {negative_image} {conformers_file} -s {shaep_output_sdf} --nStructures {n_poses} --output-file {shaep_output_txt} -j 1"
			subprocess.call(shaep_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	except Exception as e:
		printlog(f"SHAEP similarity calculation to negative image failed: {e}")

	return


def fetch_panther_poses(w_dir: Union[str, Path], n_poses: int):
	"""
	Fetches PANTHER poses from SHAEP result files and combines them into a single SDF file.

	Args:
		w_dir (Union[str, Path]): The working directory where the PANTHER folder is located.
		n_poses (int): The maximum number of poses to keep for each molecule.

	Returns:
		Union[str, None]: The path to the output SDF file if successful, None otherwise.
	"""
	panther_folder = Path(w_dir) / "panther"
	output_file = panther_folder / "panther_poses.sdf"

	if panther_folder.is_dir() and not output_file.is_file():
		try:
			all_poses = []
			shaep_results_files = list(panther_folder.glob("shaep_results_*.sdf"))

			for shaep_results in shaep_results_files:
				df = PandasTools.LoadSDF(str(shaep_results),
											molColName="Molecule",
											smilesName="SMILES",
											strictParsing=False)

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

				all_poses.append(df)

			if not all_poses:
				raise ValueError("No valid poses were found in any of the SHAEP result files.")

			# Combine all poses
			combined_poses = pd.concat(all_poses, ignore_index=True)

			# Sort by ID and then by PANTHER_Score to ensure consistent ordering
			combined_poses = combined_poses.sort_values(['ID', 'PANTHER_Score'], ascending=[True, False])

			PandasTools.WriteSDF(combined_poses,
									str(output_file),
									molColName="Molecule",
									idName="Pose ID",
									properties=list(combined_poses.columns))
			print(f"Successfully wrote combined poses to {output_file}")

		except Exception as e:
			print(f"ERROR: Failed to process PANTHER poses: {str(e)}")
			return None
		else:
			delete_files(Path(w_dir) / "panther", "panther_poses.sdf")
	return output_file
