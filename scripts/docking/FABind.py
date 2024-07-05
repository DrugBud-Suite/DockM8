import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, AllChem
from tqdm import tqdm

# Assuming similar path structure as in the provided code
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog
from scripts.utilities.utilities import delete_files


def fabind_docking(split_file: Path,
					w_dir: Path,
					protein_file: str,
					pocket_definition: dict,
					software: Path,
					n_poses: int):
	# Create necessary folders
	fabind_folder = w_dir / "fabind"
	fabind_folder.mkdir(parents=True, exist_ok=True)
	temp_files_dir = fabind_folder / "temp_files"
	temp_files_dir.mkdir(parents=True, exist_ok=True)
	save_mols_dir = temp_files_dir / "mol"
	save_mols_dir.mkdir(parents=True, exist_ok=True)

	# Create pdb directory and copy protein file
	pdb_dir = fabind_folder / "pdb"
	pdb_dir.mkdir(parents=True, exist_ok=True)
	protein_file_path = Path(protein_file)
	shutil.copy(protein_file_path, pdb_dir / protein_file_path.name)

	# Get protein name without extension
	protein_name = protein_file_path.stem

	# Prepare input files
	input_file = split_file if split_file else w_dir / "final_library.sdf"
	index_csv = fabind_folder / f"{os.path.basename(input_file).split('.')[0]}_index.csv"

	# Create index CSV from input SDF
	df = PandasTools.LoadSDF(str(input_file), molColName='ROMol', smilesName='SMILES')

	def clean_smiles(mol):
		return Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))

	df['Cleaned_SMILES'] = df['ROMol'].apply(clean_smiles)
	df['pdb_id'] = protein_name                     # Use protein name instead of placeholder
	csv_df = df[['Cleaned_SMILES', 'pdb_id', 'ID']]
	csv_df.columns = ['SMILES', 'pdb_id', 'ligand_id']
	csv_df.to_csv(index_csv, index=False)

	# Define commands
	conda_activate_cmd = "conda run -n fabind "
	preprocess_mol_cmd = (
		f"{conda_activate_cmd} python {software}/FABind/FABind_plus/fabind/inference_preprocess_mol_confs.py "
		f"--index_csv {index_csv} "
		f"--save_mols_dir {save_mols_dir} "
		f"--num_threads {int(os.cpu_count()*0.9)}")
	preprocess_protein_cmd = (
		f"{conda_activate_cmd} python {software}/FABind/FABind_plus/fabind/inference_preprocess_protein.py "
		f"--pdb_file_dir {pdb_dir} "
		f"--save_pt_dir {temp_files_dir}")
	inference_cmd = (f"{conda_activate_cmd} python {software}/FABind/FABind_plus/fabind/inference_fabind.py "
						f"--ckpt {software}/FABind/FABind_plus/ckpt/fabind_plus_best_ckpt.bin "
						f"--batch_size 8 "
						f"--post-optim "
						f"--write-mol-to-file "
						f"--sdf-output-path-post-optim {fabind_folder} "
						f"--index-csv {index_csv} "
						f"--preprocess-dir {temp_files_dir} ")

	# Execute commands
	print("Preprocessing molecules...")
	subprocess.run(preprocess_mol_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
	print("Preprocessing protein...")
	subprocess.run(preprocess_protein_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
	print("Running FABind+ inference...")
	subprocess.run(inference_cmd, shell=True)

	print("Docking completed successfully!")


def fetch_fabind_poses(w_dir: Union[str, Path]):
	"""
    Fetches FABind+ poses from the specified directory and combines them into a single SDF file.

    Args:
        w_dir (str or Path): The directory path where the FABind+ poses are located.
        protein_file (str): Path to the protein file (not used in this version, kept for compatibility).
        n_poses (int): The number of poses to fetch (not used in this implementation as FABind+ generates one pose).

    Returns:
        Path: Path to the combined poses SDF file.
    """
	fabind_folder = Path(w_dir) / "fabind"
	output_file = fabind_folder / "fabind_poses.sdf"

	if fabind_folder.is_dir() and not output_file.is_file():
		try:
			fabind_dataframes = []
			for file in tqdm(os.listdir(fabind_folder), desc="Loading FABind+ poses"):
				if file.endswith(".sdf"):
					try:
						df = PandasTools.LoadSDF(str(fabind_folder / file),
													molColName="Molecule",
													smilesName="SMILES",
													strictParsing=False)

						ligand_id = file[:-4]
						df["ID"] = ligand_id
						df["Pose ID"] = f"{ligand_id}_FABind_1"
						fabind_dataframes.append(df)
					except Exception as e:
						print(f"WARNING: Failed to load {file}: {str(e)}")
						continue

			if not fabind_dataframes:
				raise ValueError("No valid poses were loaded")

			fabind_df = pd.concat(fabind_dataframes, ignore_index=True)

		except Exception as e:
			print(f"ERROR: Failed to load or process FABind poses: {str(e)}")
			return None

		try:
			PandasTools.WriteSDF(fabind_df,
									str(output_file),
									molColName="Molecule",
									idName="Pose ID",
									properties=list(fabind_df.columns))
			print(f"Successfully wrote combined poses to {output_file}")

		except Exception as e:
			print(f"ERROR: Failed to write combined FABind poses SDF file: {str(e)}")
			return None
		else:
			delete_files(w_dir / "fabind", ["fabind_poses.sdf"])

	return output_file
