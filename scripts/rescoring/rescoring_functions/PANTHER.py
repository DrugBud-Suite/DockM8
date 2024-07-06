import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Union

import pandas as pd
from rdkit.Chem import PandasTools

# Assuming similar path structure as in the provided code
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.file_splitting import split_sdf_str
from scripts.utilities.logging import printlog
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.parallel_executor import parallel_executor
from scripts.utilities.utilities import delete_files


def PANTHER_rescoring(sdf: str, n_cpus: int, column_name: str, **kwargs):
	"""
    Performs rescoring of ligand poses using the PANTHER software package.

    Args:
        sdf (str): The path to the input SDF file.
        n_cpus (int): The number of CPUs to use for parallel execution.
        column_name (str): The name of the column in the output dataframe that will contain the rescoring results.

    Returns:
        A Pandas dataframe containing the rescoring results.
    """
	tic = time.perf_counter()
	rescoring_folder = kwargs.get("rescoring_folder")
	software = kwargs.get("software")
	protein_file = kwargs.get("protein_file")
	pocket_definition = kwargs.get("pocket_definition")
	split_files_folder = split_sdf_str(rescoring_folder / f"{column_name}_rescoring", sdf, n_cpus)
	split_files_sdfs = [split_files_folder / f for f in os.listdir(split_files_folder) if f.endswith(".sdf")]

	negative_image = generate_negative_image(rescoring_folder / f"{column_name}_rescoring",
												software,
												protein_file,
												pocket_definition)

	global panther_rescoring_splitted

	def panther_rescoring_splitted(split_file, negative_image):
		try:
			# Convert SDF to MOL2
			mol2_file = split_file.with_suffix('.mol2')

			obabel_charge_cmd = f"obabel -isdf {split_file} -O {mol2_file} --partialcharge mmff94"
			subprocess.call(obabel_charge_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
			# Check if SHAEP executable exists
			shaep_executable = software / "shaep"
			if not shaep_executable.is_file():
				printlog(
					"SHAEP executable not found. Please download SHAEP and ensure it is in DockM8's software folder. SHAEP can be found here: https://users.abo.fi/mivainio/shaep/download.php"
				)
			else:
				panther_folder = rescoring_folder / f"{column_name}_rescoring"
				# Run SHAEP
				shaep_output_sdf = panther_folder / f"{Path(split_file).stem}_{column_name}.sdf"
				shaep_output_txt = panther_folder / f"{Path(split_file).stem}_{column_name}.txt"
				shaep_cmd = f"{software}/shaep -q {negative_image} {mol2_file} -s {shaep_output_sdf} --output-file {shaep_output_txt} --noOptimization"
				subprocess.call(shaep_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

			# Clean up the temporary MOL2 file
			os.remove(mol2_file)

		except Exception as e:
			printlog(f"SHAEP similarity calculation to negative image failed: {e}")

		return shaep_output_sdf

	# Run PANTHER rescoring in parallel
	rescoring_results = parallel_executor(panther_rescoring_splitted,
											split_files_sdfs,
											n_cpus,
											negative_image=negative_image)

	# Process the results
	panther_dataframes = []
	for result_file in rescoring_results:
		if result_file and Path(result_file).is_file():
			df = PandasTools.LoadSDF(str(result_file),
										idName="Pose ID",
										molColName=None,
										includeFingerprints=False,
										embedProps=False)
			panther_dataframes.append(df)

	if not panther_dataframes:
		printlog(f"ERROR: No valid results found for {column_name} rescoring!")
		return None

	panther_rescoring_results = pd.concat(panther_dataframes)
	panther_rescoring_results = panther_rescoring_results[[
		"Pose ID", "Similarity_best", "Similarity_ESP", "Similarity_shape"]]
	panther_rescoring_results.rename(columns={
		"Similarity_best": "PANTHER", "Similarity_ESP": "PANTHER-ESP", "Similarity_shape": "PANTHER-Shape"},
										inplace=True)
	panther_rescoring_results = panther_rescoring_results[["Pose ID", column_name]]
	panther_scores_path = rescoring_folder / f"{column_name}_rescoring" / f"{column_name}_scores.csv"
	panther_rescoring_results.to_csv(panther_scores_path, index=False)
	delete_files(rescoring_folder / f"{column_name}_rescoring", f"{column_name}_scores.csv")

	toc = time.perf_counter()
	printlog(f"Rescoring with {column_name} complete in {toc - tic:0.4f}!")
	return panther_rescoring_results


def generate_negative_image(rescoring_folder, software, protein_file, pocket_definition):
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
				center = pocket_definition['center']
				lines[i] = f"5-Center(s) (-cent):: {center[0]} {center[1]} {center[2]}\n"
			elif line.startswith("9-Box radius"):
				box_size = pocket_definition['size'][0] // 2
				lines[i] = f"9-Box radius (-brad):: {box_size}\n"
		# Write the modified input file
		panther_input = rescoring_folder / "panther_input.in"
		with open(panther_input, 'w') as f:
			f.writelines(lines)
		# Run PANTHER and capture its output
		negative_image = rescoring_folder / "negative_image.mol2"
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
		printlog(f"PANTHER stdout: {stdout}")
		printlog(f"PANTHER stderr: {stderr}")
	return negative_image
