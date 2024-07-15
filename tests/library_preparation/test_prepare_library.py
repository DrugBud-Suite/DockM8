import sys
from pathlib import Path
import pytest
from rdkit.Chem import PandasTools
from scripts.library_preparation.library_preparation import prepare_library

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests"
					for p in Path(__file__).resolve().parents
					if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))


@pytest.fixture
def common_test_data():
	"""Set up common test data."""
	dockm8_path = next((p for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
	library = dockm8_path / "tests/test_files/library_preparation/library.sdf"
	software = dockm8_path / "software"
	return library, software

@pytest.fixture
def cleanup(request):
	"""Cleanup fixture to remove generated files after each test."""
	def remove_created_files():
		# Add cleanup logic if necessary
		pass
	request.addfinalizer(remove_created_files)

def test_prepare_library_protonation(common_test_data, cleanup):
	"""Test library preparation with standardization and protonation."""
	library, software = common_test_data
	n_cpus = 4  # Set a fixed number of CPUs for testing

	# Load input library
	input_df = PandasTools.LoadSDF(str(library), molColName="Molecule", idName="ID")

	# Call the function
	final_df = prepare_library(library, "GypsumDL", "GypsumDL", software, n_cpus)

	# Check if the output DataFrame is not empty
	assert not final_df.empty

	# Check if the number of molecules in the output is less than or equal to the input
	# (some molecules might fail during preparation)
	assert len(final_df) <= len(input_df)

	# Check if the output DataFrame has the expected columns
	assert set(final_df.columns) == {"Molecule", "ID"}

	# Check if all molecules in the output DataFrame are valid
	assert all(mol is not None for mol in final_df["Molecule"])

	# Check if all molecules have 3D coordinates
	assert all(mol.GetNumConformers() > 0 for mol in final_df["Molecule"])

def test_prepare_library_no_protonation(common_test_data, cleanup):
	"""Test library preparation without protonation."""
	library, software = common_test_data
	n_cpus = 4  # Set a fixed number of CPUs for testing

	# Load input library
	input_df = PandasTools.LoadSDF(str(library), molColName="Molecule", idName="ID")

	# Call the function
	final_df = prepare_library(library, "None", "GypsumDL", software, n_cpus)

	# Check if the output DataFrame is not empty
	assert not final_df.empty

	# Check if the number of molecules in the output is less than or equal to the input
	assert len(final_df) <= len(input_df)

	# Check if the output DataFrame has the expected columns
	assert set(final_df.columns) == {"Molecule", "ID"}

	# Check if all molecules in the output DataFrame are valid
	assert all(mol is not None for mol in final_df["Molecule"])

	# Check if all molecules have 3D coordinates
	assert all(mol.GetNumConformers() > 0 for mol in final_df["Molecule"])

def test_prepare_library_invalid_protonation(common_test_data, cleanup):
	"""Test library preparation with invalid protonation method."""
	library, software = common_test_data
	n_cpus = 4  # Set a fixed number of CPUs for testing

	# Check if the function raises a ValueError for an invalid protonation method
	with pytest.raises(ValueError):
		prepare_library(library, "InvalidMethod", "GypsumDL", software, n_cpus)

def test_prepare_library_invalid_conformers(common_test_data, cleanup):
	"""Test library preparation with invalid conformers method."""
	library, software = common_test_data
	n_cpus = 4  # Set a fixed number of CPUs for testing

	# Check if the function raises a ValueError for an invalid conformers method
	with pytest.raises(ValueError):
		prepare_library(library, "GypsumDL", "InvalidMethod", software, n_cpus)
