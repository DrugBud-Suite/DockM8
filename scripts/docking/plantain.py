import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm import tqdm

# Assuming similar path structure as in the provided code
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog
from scripts.utilities.utilities import delete_files


def plantain_docking(split_file: Path,
						w_dir: Path,
						protein_file: str,
						pocket_definition: Dict[str, list],
						software: Path,
						n_poses: int):
	"""
    Dock ligands using PLANTAIN.

    Args:
        split_file (Path): Path to the split file containing ligands to dock.
        w_dir (Path): Path to the working directory.
        protein_file (str): Path to the protein file.
        pocket_definition (Dict[str, list]): Dictionary containing the center and size of the pocket.
        software (Path): Path to the PLANTAIN software.
        n_poses (int): Number of poses to generate.

    Returns:
        None
    """
	# Create a folder to store the PLANTAIN results
	plantain_folder = w_dir / "plantain"
	plantain_folder.mkdir(parents=True, exist_ok=True)

	# Prepare input files
	input_file = split_file if split_file else w_dir / "final_library.sdf"
	output_file = plantain_folder / f"{os.path.basename(input_file).split('.')[0]}_plantain_poses.sdf"

	# Convert SDF to SMILES
	smiles_file = plantain_folder / f"{os.path.basename(input_file).split('.')[0]}_compounds.smi"
	mols = Chem.SDMolSupplier(str(input_file))
	with open(smiles_file, 'w') as f:
		for mol in mols:
			if mol is not None:
				smi = Chem.MolToSmiles(mol)
				name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
				f.write(f"{smi} {name}\n")

	# Prepare the pocket definition
	protein_pocket = str(protein_file).replace(".pdb", "_pocket.pdb")

	# Construct the PLANTAIN command
	plantain_cmd = (f"cd {software}/plantain && python ./inference.py "
					f"{smiles_file} {protein_file} "
					f"--out {plantain_folder}/predictions "
					f"--num_workers 1 --no_gpu")

	try:
		# Execute the PLANTAIN command
		subprocess.call(plantain_cmd, shell=True)

		# Combine all output SDF files into one
		writer = Chem.SDWriter(str(output_file))
		for sdf_file in plantain_folder.glob("predictions/*.sdf"):
			for mol in Chem.SDMolSupplier(str(sdf_file)):
				if mol is not None:
					writer.write(mol)
		writer.close()

		printlog(f"PLANTAIN docking completed. Results saved to {output_file}")
	except Exception as e:
		printlog(f"PLANTAIN docking failed: {e}")

	return


def fetch_plantain_poses(w_dir: Union[str, Path], n_poses: int):
	"""
    Fetches PLANTAIN poses from the specified directory and combines them into a single SDF file.

    Args:
        w_dir (Union[str, Path]): The directory path where the PLANTAIN poses are located.
        n_poses (int): The number of poses to fetch for each ligand.

    Returns:
        Path: Path to the combined poses SDF file.
    """
	plantain_folder = Path(w_dir) / "plantain"
	output_file = plantain_folder / "plantain_poses.sdf"

	if plantain_folder.is_dir() and not output_file.is_file():
		try:
			all_poses = []
			for sdf_file in tqdm(list(plantain_folder.glob("*_plantain_poses.sdf")), desc="Loading PLANTAIN poses"):
				df = PandasTools.LoadSDF(str(sdf_file),
											molColName="Molecule",
											smilesName="SMILES",
											idName='ID',
											strictParsing=False)

				# Add ranking based on the order of poses in the file
				df['PLANTAIN_Rank'] = range(1, len(df) + 1)

				# Keep only the top n_poses for each molecule
				df = df[df['PLANTAIN_Rank'] <= n_poses]

				# Create Pose ID
				df['Pose ID'] = df.apply(lambda row: f"{row['ID']}_PLANTAIN_{int(row['PLANTAIN_Rank'])}", axis=1)

				all_poses.append(df)

			if not all_poses:
				raise ValueError("No valid poses were found in any of the PLANTAIN output files.")

			# Combine all poses
			combined_poses = pd.concat(all_poses, ignore_index=True)

			# Sort by ID and then by PLANTAIN_Rank to ensure consistent ordering
			combined_poses = combined_poses.sort_values(['ID', 'PLANTAIN_Rank'], ascending=[True, True])

			PandasTools.WriteSDF(combined_poses,
									str(output_file),
									molColName="Molecule",
									idName="Pose ID",
									properties=list(combined_poses.columns))

		except Exception as e:
			printlog(f"ERROR: Failed to process PLANTAIN poses: {str(e)}")
			return None
		else:
			delete_files(plantain_folder, ["plantain_poses.sdf"])

	return output_file
