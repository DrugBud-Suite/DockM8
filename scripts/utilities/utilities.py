import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import List, Union

import pandas as pd
from rdkit import Chem

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_molecule(molecule_file):
	"""Load a molecule from a file.
    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format '.mol2', '.mol', '.sdf',
        '.pdbqt', or '.pdb'.
    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the loaded molecule.
    """
	if molecule_file.endswith(".mol2"):
		mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
	if molecule_file.endswith(".mol"):
		mol = Chem.MolFromMolFile(molecule_file, sanitize=False, removeHs=False)
	elif molecule_file.endswith(".sdf"):
		supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
		mol = supplier[0]
	elif molecule_file.endswith(".pdbqt"):
		with open(molecule_file) as f:
			pdbqt_data = f.readlines()
		pdb_block = ""
		for line in pdbqt_data:
			pdb_block += "{}\n".format(line[:66])
		mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
	elif molecule_file.endswith(".pdb"):
		mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
	else:
		return ValueError(
			f"Expect the format of the molecule_file to be one of .mol2, .mol, .sdf, .pdbqt and .pdb, got {molecule_file}"
		)
	return mol


def delete_files(folder_path: str, save_file: Union[str, List[str]]) -> None:
	"""
    Deletes all files in a folder except for specified save files and patterns.

    Args:
        folder_path (str): The path to the folder to delete files from.
        save_file (Union[str, List[str]]): The name(s) or pattern(s) of the file(s) to save.

    Returns:
        None
    """
	folder = Path(folder_path)
	if isinstance(save_file, str):
		save_file = [save_file]

	# Expand the save list to include files matching patterns
	expanded_save_files = set()
	for pattern in save_file:
		expanded_save_files.update(folder.glob(pattern))

	for item in folder.iterdir():
		if item.is_file() and item not in expanded_save_files:
			item.unlink()
		elif item.is_dir():
			delete_files(item, save_file)
			if not any(item.iterdir()) and item not in expanded_save_files:
				item.rmdir()


def str2bool(v):
	"""
    Converts a string representation of a boolean to a boolean value.
    """
	if isinstance(v, bool):
		return v
	if v.lower() in ("yes", "true", "t", "y", "1", "True"):
		return True
	elif v.lower() in ("no", "false", "f", "n", "0", "False"):
		return False
	else:
		raise argparse.ArgumentTypeError("Boolean value expected.")


import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem


def parallel_SDF_loader(sdf_path: Path,
						molColName: str,
						idName: str,
						n_cpus: Optional[int] = None,
						SMILES: Optional[str] = None) -> pd.DataFrame:
	"""
    Loads an SDF file in parallel using ThreadPoolExecutor.

    Args:
        sdf_path (Path): The path to the SDF file.
        molColName (str): The name of the molecule column in the output DataFrame.
        idName (str): The name of the ID column in the output DataFrame.
        n_cpus (int, optional): The number of CPUs to use. Defaults to (CPU count - 2).
        SMILES (str, optional): SMILES string (unused in current implementation).

    Returns:
        DataFrame: The loaded SDF file as a DataFrame.
    """
	if n_cpus is None:
		n_cpus = max(1, int(os.cpu_count() * 0.9))

	def process_molecule(mol):
		if mol is None:
			return None
		mol_props = {"Pose ID": mol.GetProp("_Name")}
		mol_props.update({prop: mol.GetProp(prop) for prop in mol.GetPropNames()})
		mol_props["Molecule"] = mol
		return mol_props

	try:
		supplier = Chem.MultithreadedSDMolSupplier(str(sdf_path), numWriterThreads=n_cpus)

		with ThreadPoolExecutor(max_workers=n_cpus) as executor:
			future_to_mol = {executor.submit(process_molecule, mol): mol for mol in supplier}
			data = [future.result() for future in as_completed(future_to_mol) if future.result() is not None]

		df = pd.DataFrame(data)
		df = df.drop(columns=["mol_cond"], errors="ignore")

		# Convert numeric columns
		for col in df.columns:
			if col not in [idName, molColName, "ID", "Pose ID", "SMILES", SMILES]:
				df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

		return df

	except Exception as e:
		print(f"Error occurred during loading of SDF file: {str(e)}")
		return pd.DataFrame()      # Return an empty DataFrame instead of None
