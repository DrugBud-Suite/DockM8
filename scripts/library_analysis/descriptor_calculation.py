import pandas as pd
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
import os
from pathlib import Path
import sys

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.parallel_executor import parallel_executor


def calculate_single_property(args):
	"""
	Calculate a specific property for a given molecule.

	Args:
		args (tuple): A tuple containing the molecule and the property name.

	Returns:
		float: The calculated value of the specified property.

	Raises:
		ValueError: If an invalid property name is provided.
	"""
	mol, property_name = args
	if property_name == 'MW':
		return Descriptors.MolWt(mol)
	elif property_name == 'TPSA':
		return Descriptors.TPSA(mol)
	elif property_name == 'HBA':
		return rdMolDescriptors.CalcNumHBA(mol)
	elif property_name == 'HBD':
		return rdMolDescriptors.CalcNumHBD(mol)
	elif property_name == 'Rotatable Bonds':
		return Descriptors.NumRotatableBonds(mol)
	elif property_name == 'QED':
		return QED.qed(mol)
	elif property_name == 'sp3 percentage':
		return rdMolDescriptors.CalcFractionCSP3(mol)
	elif property_name == 'Ring Count':
		return rdMolDescriptors.CalcNumRings(mol)
	else:
		raise ValueError("Invalid property name provided.")


def calculate_properties(df: pd.DataFrame, properties: list, n_cpus=int(os.cpu_count() * 0.9)) -> pd.DataFrame:
	"""
	Calculate properties for molecules in a DataFrame.

	Args:
		df (pandas.DataFrame): The DataFrame containing the molecules.
		properties (list): A list of property names to calculate.
		n_cpus (int, optional): The number of CPUs to use for parallel execution. Defaults to 4.

	Returns:
		pandas.DataFrame: The DataFrame with the calculated properties added as columns.

	"""
	for property_name in properties:
		if property_name in df.columns:
			continue

		property_args = [(mol, property_name) for mol in df['Molecule']]
		results = parallel_executor(calculate_single_property,
									property_args,
									n_cpus=n_cpus,
									job_manager="concurrent_process",
									display_name=f"Calculating {property_name}")

		df[property_name] = results
		df[property_name] = df[property_name].astype(float)

	return df
