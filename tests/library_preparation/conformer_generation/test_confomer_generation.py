import os
import sys
from pathlib import Path
import pytest
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.library_preparation.conformer_generation.confgen_GypsumDL import generate_conformers_GypsumDL
from scripts.library_preparation.conformer_generation.confgen_RDKit import generate_conformers_RDKit


@pytest.fixture
def common_test_data():
	"""Set up common test data."""
	dockm8_path = next((p for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
	library = dockm8_path / "tests/test_files/library_preparation/library.sdf"
	software = dockm8_path / "software"
	return library, software


def test_generate_conformers_GypsumDL(common_test_data):
	"""Test generate_conformers_GypsumDL function."""
	library, software = common_test_data
	n_cpus = 4                  # Set a fixed number of CPUs for testing

	# Load input library
	input_df = PandasTools.LoadSDF(str(library), molColName="Molecule", idName="ID")

	# Call the function
	output_df = generate_conformers_GypsumDL(input_df, software, n_cpus)

	# Check if the output DataFrame is not empty
	assert not output_df.empty

	# Check if the number of molecules in the input and output DataFrames are the same
	assert len(input_df) == len(output_df)

	# Check if the output DataFrame has the expected columns
	assert set(output_df.columns) == {"Molecule", "ID"}

	# Check if all molecules in the output DataFrame have 3D coordinates
	assert all(mol.GetNumConformers() > 0 for mol in output_df["Molecule"])

	# Check if the IDs in the input and output DataFrames match
	assert set(input_df["ID"]) == set(output_df["ID"])


def test_generate_conformers_RDKit(common_test_data):
	"""Test generate_conformers_RDKit function."""
	library, software = common_test_data
	n_cpus = 4                  # Set a fixed number of CPUs for testing
	forcefield = 'MMFF'         # Choose either 'MMFF' or 'UFF'

	# Load input library
	input_df = PandasTools.LoadSDF(str(library), molColName="Molecule", idName="ID")

	# Call the function
	output_df = generate_conformers_RDKit(input_df, n_cpus, forcefield)

	# Check if the output DataFrame is not empty
	assert not output_df.empty

	# Check if the number of molecules in the output is less than or equal to the input
	# (some molecules might fail conformer generation)
	assert len(output_df) <= len(input_df)

	# Check if all molecules in the output DataFrame have 3D conformers
	assert all(mol.GetConformer().Is3D() for mol in output_df["Molecule"])

	# Check if the IDs in the output DataFrame are a subset of the input DataFrame
	assert set(output_df["ID"]).issubset(set(input_df["ID"]))
