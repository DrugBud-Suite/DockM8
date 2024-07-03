import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from molvs import Standardizer, tautomer
from rdkit import Chem

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog
from scripts.utilities.parallel_executor import parallel_executor


class StandardizationError(Exception):

	"""Custom exception for standardization errors."""
	pass


def standardize_molecule(molecule: Chem.Mol,
							remove_salts: bool = True,
							standardize_tautomers: bool = True) -> Tuple[Optional[Chem.Mol], Optional[str]]:
	"""Standardize a single molecule using MolVS.

	Args:
		molecule (Chem.Mol): The molecule to be standardized.
		remove_salts (bool, optional): Whether to remove salts from the molecule. Defaults to True.
		standardize_tautomers (bool, optional): Whether to standardize tautomers of the molecule. Defaults to True.

	Returns:
		Tuple[Optional[Chem.Mol], Optional[str]]: A tuple containing the standardized molecule and an error message, if any.

	"""
	s = Standardizer()

	try:
		if remove_salts:
			std_molecule = s.fragment_parent(molecule)

		std_molecule = s.standardize(std_molecule)

		if standardize_tautomers:
			std_molecule = tautomer.canonicalize(std_molecule)

		return std_molecule, None
	except Exception as e:
		return molecule, str(StandardizationError(f"Standardization failed: {str(e)}"))


def standardize_ids(df: pd.DataFrame, id_column: str = 'ID') -> pd.DataFrame:
	"""
	Standardizes the IDs in the given DataFrame.

	Args:
		df (pd.DataFrame): The DataFrame containing the IDs to be standardized.
		id_column (str, optional): The name of the column containing the IDs. Defaults to 'ID'.

	Returns:
		pd.DataFrame: The DataFrame with standardized IDs.

	"""
	if df[id_column].isnull().all():
		df[id_column] = [f"DOCKM8-{i+1}" for i in range(len(df))]
	else:
		df[id_column] = df[id_column].astype(str).apply(lambda x: f"DOCKM8-{x}" if x.isdigit() else x)
		df[id_column] = df[id_column].apply(lambda x: re.sub(r"[^a-zA-Z0-9-]", "", x))
	return df


def standardize_library(df: pd.DataFrame,
						id_column: str = 'ID',
						smiles_column: str = 'SMILES',
						remove_salts: bool = True,
						standardize_tautomers: bool = True,
						standardize_ids_flag: bool = True,
						n_cpus: int = int(os.cpu_count() * 0.9)) -> pd.DataFrame:
	"""
    Standardizes a docking library using MolVS.

    Args:
        df (pd.DataFrame): Input DataFrame containing the docking library.
        id_column (str): Column name containing the compound IDs.
        smiles_column (str): Column name containing the SMILES strings.
        remove_salts (bool): Whether to remove salts from the molecules.
        standardize_tautomers (bool): Whether to standardize tautomers.
        standardize_ids_flag (bool): Whether to standardize IDs.
        n_cpus (int): Number of CPUs for parallel processing.

    Returns:
        Tuple[pd.DataFrame, List[str]]: Standardized DataFrame and list of error messages.
    """
	printlog("Standardizing docking library using MolVS...")

	# Convert SMILES strings to RDKit molecules
	if 'Molecule' not in df.columns:
		df['Molecule'] = df[smiles_column].apply(Chem.MolFromSmiles)

	# Standardize IDs if the flag is set
	if standardize_ids_flag:
		df = standardize_ids(df, id_column)

	# Standardize the molecules using MolVS

	results = parallel_executor(standardize_molecule,
								df['Molecule'].tolist(),
								n_cpus,
								'concurrent_process',
								remove_salts=remove_salts,
								standardize_tautomers=standardize_tautomers)

	# Separate molecules and error messages
	standardized_molecules, error_messages = zip(*results)
	df['Molecule'] = standardized_molecules

	# Remove entries where standardization failed
	n_cpds_start = len(df)
	df = df[df['Molecule'].notnull()]
	n_cpds_end = len(df)

	printlog(
		f"Standardization finished: Started with {n_cpds_start}, ended with {n_cpds_end}: {n_cpds_start - n_cpds_end} compounds lost"
	)

	# Update SMILES column with standardized molecules
	df[smiles_column] = df['Molecule'].apply(Chem.MolToSmiles)

	return df
