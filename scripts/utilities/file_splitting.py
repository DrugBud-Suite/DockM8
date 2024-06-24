import math
import os
from pathlib import Path
import sys
import warnings
import pandas as pd
from rdkit.Chem import PandasTools
from tqdm import tqdm

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def split_sdf(dir, sdf_file, n_cpus):
	"""
    Split an SDF file into multiple smaller SDF files, each containing a subset of the original compounds.

    Args:
        dir (str): The directory where the split SDF files will be saved.
        sdf_file (str): The path to the original SDF file to be split.
        n_cpus (int): The number of CPUs to use for the splitting process.

    Returns:
        Path: The path to the directory containing the split SDF files.
    """
	sdf_file_name = Path(sdf_file).name.replace(".sdf", "")
	split_files_folder = Path(dir) / f"split_{sdf_file_name}"
	split_files_folder.mkdir(parents=True, exist_ok=True)
	for file in split_files_folder.iterdir():
		file.unlink()
	df = PandasTools.LoadSDF(str(sdf_file), molColName="Molecule", idName="ID", includeFingerprints=False)
	compounds_per_core = math.ceil(len(df["ID"]) / (n_cpus*2))
	used_ids = set()
	file_counter = 1
	for i in range(0, len(df), compounds_per_core):
		chunk = df[i:i + compounds_per_core]
		chunk = chunk[~chunk["ID"].isin(used_ids)]
		used_ids.update(set(chunk["ID"]))
		output_file = split_files_folder / f"split_{file_counter}.sdf"
		PandasTools.WriteSDF(chunk, str(output_file), molColName="Molecule", idName="ID")
		file_counter += 1
	return split_files_folder


def split_sdf_str(dir, sdf_file, n_cpus):
	"""
	Split an SDF file into multiple smaller SDF files based on the number of compounds.

	Args:
		dir (str): The directory where the split SDF files will be saved.
		sdf_file (str): The path to the input SDF file.
		n_cpus (int): The number of CPUs to use for splitting.

	Returns:
		Path: The path to the folder containing the split SDF files.
	"""
	sdf_file_name = Path(sdf_file).name.replace(".sdf", "")
	split_files_folder = Path(dir) / f"split_{sdf_file_name}"
	split_files_folder.mkdir(parents=True, exist_ok=True)

	with open(sdf_file, "r") as infile:
		sdf_lines = infile.readlines()

	total_compounds = sdf_lines.count("$$$$\n")

	if total_compounds > 100000:
		n = max(1, math.ceil(total_compounds // n_cpus // 8))
	else:
		n = max(1, math.ceil(total_compounds // n_cpus // 2))

	compound_count = 0
	file_index = 1
	current_compound_lines = []

	for line in sdf_lines:
		current_compound_lines.append(line)

		if line.startswith("$$$$"):
			compound_count += 1

			if compound_count % n == 0:
				output_file = split_files_folder / f"split_{file_index}.sdf"
				with open(output_file, "w") as outfile:
					outfile.writelines(current_compound_lines)
				current_compound_lines = []
				file_index += 1

	# Write the remaining compounds to the last file
	if current_compound_lines:
		output_file = split_files_folder / f"split_{file_index}.sdf"
		with open(output_file, "w") as outfile:
			outfile.writelines(current_compound_lines)

	return split_files_folder


def split_sdf_single(dir, sdf_file):
	"""
    Split a single SDF file into multiple SDF files, each containing one compound.

    Args:
    - dir (str): The directory where the split SDF files will be saved.
    - sdf_file (str): The path to the input SDF file.

    Returns:
    - split_files_folder (Path): The path to the directory containing the split SDF files.
    """
	sdf_file_name = Path(sdf_file).name.replace(".sdf", "")
	split_files_folder = Path(dir) / f"split_{sdf_file_name}"
	split_files_folder.mkdir(exist_ok=True)
	for file in split_files_folder.iterdir():
		file.unlink()
	df = PandasTools.LoadSDF(str(sdf_file), molColName="Molecule", idName="ID", includeFingerprints=False)
	for i, row in tqdm(df.iterrows(), total=len(df), desc="Splitting SDF file"):
		# Extract compound information from the row
		compound = row["Molecule"]
		compound_id = row["ID"]
		# Create a new DataFrame with a single compound
		compound_df = pd.DataFrame({"Molecule": [compound], "ID": [compound_id]})
		# Output file path
		output_file = split_files_folder / f"split_{i + 1}.sdf"
		# Write the single compound DataFrame to an SDF file
		PandasTools.WriteSDF(compound_df, str(output_file), molColName="Molecule", idName="ID")
	print(f"Split SDF file into {len(df)} files, each containing 1 compound")
	return split_files_folder


from pathlib import Path


def split_sdf_single_str(dir, sdf_file):
	"""
	Split an SDF file into individual compounds and save them as separate files.

	Args:
		dir (str): The directory where the split files will be saved.
		sdf_file (str): The path to the input SDF file.

	Returns:
		pathlib.Path: The path to the folder containing the split files.
	"""
	sdf_file_name = Path(sdf_file).name.replace(".sdf", "")
	split_files_folder = Path(dir) / f"split_{sdf_file_name}"
	split_files_folder.mkdir(parents=True, exist_ok=True)

	with open(sdf_file, "r") as infile:
		sdf_lines = infile.readlines()

	compound_count = 0
	current_compound_lines = []

	for line in sdf_lines:
		current_compound_lines.append(line)

		if line.startswith("$$$$"):
			compound_count += 1

			output_file = split_files_folder / f"split_{compound_count}.sdf"
			with open(output_file, "w") as outfile:
				outfile.writelines(current_compound_lines)
			current_compound_lines = []

	# Write the remaining compounds to the last file
	if current_compound_lines:
		compound_count += 1
		output_file = split_files_folder / f"split_{compound_count}.sdf"
		with open(output_file, "w") as outfile:
			outfile.writelines(current_compound_lines)

	return split_files_folder


def split_pdbqt_str(file):
	"""
	Splits a PDBQT file into separate models and saves each model as a separate file.

	Args:
		file (str): The path to the PDBQT file.

	Returns:
		None
	"""
	models = []
	current_model = []
	with open(file, "r") as f:
		lines = f.readlines()
	for line in lines:
		current_model.append(line)
		if line.startswith("ENDMDL"):
			models.append(current_model)
			current_model = []
	for i, model in enumerate(models):
		for line in model:
			if line.startswith("MODEL"):
				model_number = int(line.split()[-1])
				break
		output_filename = file.with_name(f"{file.stem}_{model_number}.pdbqt")
		with open(output_filename, "w") as output_file:
			output_file.writelines(model)
	os.remove(file)
