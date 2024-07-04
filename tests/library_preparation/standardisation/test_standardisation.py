import sys
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.library_preparation.standardisation.standardise import standardize_library


@pytest.fixture
def common_test_data():
	"""Set up common test data."""
	dockm8_path = next((p for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
	library = dockm8_path / "tests/test_files/library_preparation/library.sdf"
	return library


def test_standardize_library(common_test_data):
	"""Test standardize_library function."""
	library = common_test_data
	n_cpus = 4                  # Set a fixed number of CPUs for testing

	# Load input library
	input_df = PandasTools.LoadSDF(str(library), molColName="Molecule", idName="ID")

	# Call the function
	standardized_df = standardize_library(input_df, n_cpus=n_cpus)

	# Check if the output DataFrame is not empty
	assert not standardized_df.empty

	# Check if the number of molecules in the output is less than or equal to the input
	# (some molecules might fail standardization)
	assert len(standardized_df) <= len(input_df)

	# Check if all molecules in the output DataFrame are valid
	assert all(mol is not None for mol in standardized_df["Molecule"])

	# Check if there are no duplicate IDs
	assert len(standardized_df["ID"].unique()) == len(standardized_df)

	# Check if SMILES strings are updated
	assert all(
		Chem.MolToSmiles(mol) == smiles for mol, smiles in zip(standardized_df["Molecule"], standardized_df["SMILES"]))
