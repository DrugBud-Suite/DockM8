import pytest
import os
import sys
from pathlib import Path

from Bio.PDB import PDBParser

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.minimization.minimization import minimize_receptor


@pytest.fixture
def common_test_data():
	"""Set up common test data."""
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	input_pdb_file = dockm8_path / "tests/test_files/protein_preparation/1fvv_p.pdb"
	return input_pdb_file


def test_minimize_receptor(common_test_data):
	# Define the input receptor file
	input_pdb_file = common_test_data

	# Call the minimize_receptor function
	minimized_receptor_file = minimize_receptor(input_pdb_file)

	# Assert that the minimized receptor file exists
	assert minimized_receptor_file.exists()
	parser = PDBParser()
	structure = parser.get_structure("protein", str(minimized_receptor_file))
	assert structure is not None
	os.unlink(minimized_receptor_file) if os.path.exists(minimized_receptor_file) else None
