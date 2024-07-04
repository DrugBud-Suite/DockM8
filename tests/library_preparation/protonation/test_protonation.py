import os
import sys
from pathlib import Path

import pytest
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests"
					for p in Path(__file__).resolve().parents
					if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.library_preparation.protonation.protgen_GypsumDL import protonate_GypsumDL


@pytest.fixture
def common_test_data():
	"""Set up common test data."""
	dockm8_path = next((p for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
	library = dockm8_path / "tests/test_files/library_preparation/library.sdf"
	software = dockm8_path / "software"
	return library, software


def test_protonate_GypsumDL(common_test_data):
	"""Test protonate_GypsumDL function."""
	library, software = common_test_data
	n_cpus = 4                  # Set a fixed number of CPUs for testing

	# Load input library
	input_df = PandasTools.LoadSDF(str(library), molColName="Molecule", idName="ID")

	# Call the function
	output_df = protonate_GypsumDL(input_df, software, n_cpus)

	# Check if the output DataFrame is not empty
	assert not output_df.empty

	# Check if the number of molecules in the output matches the input
	assert len(output_df) == len(input_df)

	# Check if the output DataFrame has the expected columns
	assert set(output_df.columns) == {"Molecule", "ID"}

	# Check if all molecules in the output DataFrame are valid
	assert all(mol is not None for mol in output_df["Molecule"])

	# Check if the IDs in the input and output DataFrames match
	assert set(input_df["ID"]) == set(output_df["ID"])

	# Check if any molecules have changed (protonation should modify some molecules)
	assert any(input_mol.GetNumAtoms() != output_mol.GetNumAtoms() for input_mol,
				output_mol in zip(input_df["Molecule"], output_df["Molecule"]))
