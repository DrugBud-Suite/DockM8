import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from meeko import PDBQTMolecule, RDKitMolCreate
from rdkit.Chem import PandasTools
from tqdm import tqdm

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.file_splitting import split_pdbqt_str
from scripts.utilities.molecule_conversion import convert_molecules
from scripts.utilities.utilities import delete_files
from scripts.utilities.logging import printlog

def qvina2_docking(split_file: Path,
	w_dir: Path,
	protein_file: Path,
	pocket_definition: Dict[str, list],
	software: Path,
	exhaustiveness: int,
	n_poses: int,
	):
	"""
    Perform docking using the QVINA2 software for either a library of molecules or split files.

    Args:
        w_dir (Path): Working directory for docking operations.
        protein_file (Path): Path to the protein file in PDB or pdbqt format.
        pocket_definition (dict): Dictionary containing the center and size of the docking pocket.
        software (Path): Path to the QVINA2 software folder.
        exhaustiveness (int): Level of exhaustiveness for the docking search.
        n_poses (int): Number of poses to generate for each ligand.
        split_file (Path, optional): Path to the split file containing the ligands to dock. If None, uses library.

    Returns:
        Path: Path to the combined docking results file in .sdf format.
    """
	qvina2_folder = w_dir / "qvina2"
	results_folder = qvina2_folder / Path(split_file).stem / "docked"
	if split_file:
		input_file = split_file
		pdbqt_folder = qvina2_folder / Path(split_file).stem / "pdbqt_files"
	else:
		input_file = w_dir / "final_library.sdf"
		pdbqt_folder = qvina2_folder / "pdbqt_files"

	pdbqt_folder.mkdir(parents=True, exist_ok=True)
	results_folder.mkdir(parents=True, exist_ok=True)

	# Convert molecules to pdbqt format
	try:
		convert_molecules(input_file, str(pdbqt_folder), "sdf", "pdbqt")
	except Exception as e:
		print("Failed to convert sdf file to .pdbqt")
		print(e)

	protein_file_pdbqt = convert_molecules(protein_file,
		protein_file.with_suffix(".pdbqt"),
		"pdb",
		"pdbqt")
	log = qvina2_folder / f"{os.path.basename(split_file).split('.')[0]}_qvina2.log"
	# Dock each ligand using QVINA2
	for pdbqt_file in pdbqt_folder.glob("*.pdbqt"):
		output_file = results_folder / (pdbqt_file.stem + "_QVINA2.pdbqt")
		qvina2_cmd = (f"{software / 'qvina2.1'}"
			f" --receptor {protein_file_pdbqt}"
			f" --ligand {pdbqt_file}"
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
		subprocess.call(qvina2_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
			)

	qvina2_docking_results = qvina2_folder / (Path(input_file).stem + "_qvina2.sdf")
	# Process the docked poses
	try:
		for file in results_folder.glob("*.pdbqt"):
			split_pdbqt_str(file)
		qvina2_poses = pd.DataFrame(columns=["Pose ID", "Molecule", "QVINA2_Affinity"])
		for pose_file in results_folder.glob("*.pdbqt"):
			pdbqt_mol = PDBQTMolecule.from_file(pose_file, name=pose_file.stem, skip_typing=True)
			rdkit_mol = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
			# Extracting QVINA2_Affinity from the file
			with open(pose_file) as file:
				affinity = next(line.split()[3] for line in file if "REMARK VINA RESULT:" in line)
			# Use loc to append a new row to the DataFrame
			qvina2_poses.loc[qvina2_poses.shape[0]] = {
				"Pose ID": pose_file.stem,
				"Molecule": rdkit_mol[0],
				"QVINA2_Affinity": affinity,
				"ID": pose_file.stem.split("_")[0], }
		PandasTools.WriteSDF(qvina2_poses,
			str(qvina2_docking_results),
			molColName="Molecule",
			idName="Pose ID",
			properties=list(qvina2_poses.columns))

	except Exception as e:
		printlog("ERROR: Failed to combine QVINA2 SDF file!")
		printlog(e)
	else:
		shutil.rmtree(qvina2_folder / Path(input_file).stem, ignore_errors=True)
	return qvina2_docking_results


def fetch_qvina2_poses(w_dir: Union[str, Path], *args):
	"""
    Fetches QVINA2 poses from the specified directory and combines them into a single SDF file.

    Args:
        w_dir (str or Path): The directory path where the QVINA2 poses are located.

    Returns:
        Path: The path to the combined QVINA2 poses SDF file.
    """
	if (w_dir / "qvina2").is_dir() and not (w_dir / "qvina2" / "qvina2_poses.sdf").is_file():
		try:
			qvina2_dataframes = []
			for file in tqdm(os.listdir(w_dir / "qvina2"), desc="Loading QVINA2 poses"):
				if file.startswith("split") or file.startswith("final_library") and file.endswith(".sdf"):
					df = PandasTools.LoadSDF(str(w_dir / "qvina2" / file),
						idName="Pose ID",
						molColName="Molecule",
						strictParsing=False)
					qvina2_dataframes.append(df)
			qvina2_df = pd.concat(qvina2_dataframes)
			qvina2_df["ID"] = qvina2_df["Pose ID"].apply(lambda x: x.split("_")[0])
		except Exception as e:
			printlog("ERROR: Failed to Load QVINA2 poses SDF file!")
			printlog(e)
		try:
			PandasTools.WriteSDF(qvina2_df,
				str(w_dir / "qvina2" / "qvina2_poses.sdf"),
				molColName="Molecule",
				idName="Pose ID",
				properties=list(qvina2_df.columns))

		except Exception as e:
			printlog("ERROR: Failed to write combined QVINA2 poses SDF file!")
			printlog(e)
		else:
			delete_files(w_dir / "qvina2", ["qvina2_poses.sdf", "*.log"])
	return w_dir / "qvina2" / "qvina2_poses.sdf"
