import os
import sys
from pathlib import Path
import pytest
from Bio.PDB import PDBParser

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.protein import Protein, ProteinSource


@pytest.fixture
def common_test_data():
	"""Set up common test data."""
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	input_pdb_file = dockm8_path / "tests/test_files/protein_preparation/1fvv_p.pdb"
	output_dir = dockm8_path / "tests/test_files/protein_preparation/"
	return input_pdb_file, output_dir


@pytest.fixture
def cleanup(request):
	"""Cleanup fixture to remove generated files after each test."""

	def remove_created_files():
		output_dir = dockm8_path / "tests/test_files/protein_preparation/"
		for file in output_dir.iterdir():
			if file.name.startswith(("prepared_", "2O1X", "Q221Q3")) or file.name.endswith(
				("_fixed.pdb", "_protonated.pdb", "_minimized.pdb", "_output.pdb", "_prepared.pdb")):
				file.unlink()

	request.addfinalizer(remove_created_files)


def test_protein_initialization_pdb(common_test_data, cleanup):
	"""Test initializing a Protein object with a PDB ID."""
	protein = Protein("2o1x")
	assert protein.identifier == "2o1x"
	assert protein.source == ProteinSource.PDB
	assert protein.pdb_file is not None
	assert protein.pdb_file.exists()
	assert protein.structure is not None


def test_protein_initialization_alphafold(common_test_data, cleanup):
	"""Test initializing a Protein object with a UniProt ID (AlphaFold)."""
	protein = Protein("Q221Q3")
	assert protein.identifier == "Q221Q3"
	assert protein.source == ProteinSource.ALPHAFOLD
	assert protein.pdb_file is not None
	assert protein.pdb_file.exists()
	assert protein.structure is not None


def test_protein_initialization_local_file(common_test_data, cleanup):
	"""Test initializing a Protein object with a local PDB file."""
	input_pdb_file, _ = common_test_data
	protein = Protein(str(input_pdb_file))
	assert protein.identifier == str(input_pdb_file)
	assert protein.source == ProteinSource.LOCAL
	assert protein.pdb_file == input_pdb_file
	assert protein.structure is not None


def test_protein_fix_structure(common_test_data, cleanup):
	"""Test fixing the protein structure."""
	input_pdb_file, _ = common_test_data
	protein = Protein(str(input_pdb_file))
	original_file = protein.pdb_file
	protein.fix_structure()
	assert protein.pdb_file != original_file
	assert protein.pdb_file.name.endswith("_fixed.pdb")
	assert protein.pdb_file.exists()


def test_protein_protonate(common_test_data, cleanup):
	"""Test protonating the protein structure."""
	input_pdb_file, _ = common_test_data
	protein = Protein(str(input_pdb_file))
	original_file = protein.pdb_file
	protein.protonate()
	assert protein.pdb_file != original_file
	assert protein.pdb_file.name.endswith("_protonated.pdb")
	assert protein.pdb_file.exists()


def test_protein_minimize(common_test_data, cleanup):
	"""Test minimizing the protein structure."""
	input_pdb_file, _ = common_test_data
	protein = Protein(str(input_pdb_file))
	original_file = protein.pdb_file
	protein.minimize()
	assert protein.pdb_file != original_file
	assert protein.pdb_file.name.endswith("_minimized.pdb")
	assert protein.pdb_file.exists()


def test_protein_prepare(common_test_data, cleanup):
	"""Test preparing the protein structure."""
	input_pdb_file, _ = common_test_data
	protein = Protein(str(input_pdb_file))
	protein.prepare_protein()
	assert protein.is_prepared
	assert protein.pdb_file.name.endswith("_prepared.pdb")
	assert protein.pdb_file.exists()


def test_protein_analyze_structure(common_test_data, cleanup):
	"""Test analyzing the protein structure."""
	input_pdb_file, _ = common_test_data
	protein = Protein(str(input_pdb_file))
	analysis = protein.analyze_structure()
	assert "chains" in analysis
	assert "residues" in analysis
	assert "atoms" in analysis


def test_protein_get_best_chain(cleanup):
	"""Test getting the best chain using EDIA scores."""
	protein = Protein("2o1x")
	best_chain = protein.get_best_chain()
	assert isinstance(best_chain, str)
	assert len(best_chain) == 1


def test_protein_to_pdb(common_test_data, cleanup):
	"""Test saving the protein structure to a PDB file."""
	input_pdb_file, output_dir = common_test_data
	protein = Protein(str(input_pdb_file))
	output_file = output_dir / "test_output.pdb"
	protein.to_pdb(output_file)
	assert output_file.exists()
	parser = PDBParser()
	structure = parser.get_structure("test", str(output_file))
	assert structure is not None


def test_protein_get_sequence(common_test_data, cleanup):
	"""Test getting the protein sequence."""
	input_pdb_file, _ = common_test_data
	protein = Protein(str(input_pdb_file))
	sequence = protein.get_sequence()
	assert isinstance(sequence, str)
	assert len(sequence) > 0


def test_protein_invalid_input():
	"""Test initializing a Protein object with invalid input."""
	with pytest.raises(ValueError):
		Protein("invalid_input")


def test_protein_create_temp_dir(common_test_data, cleanup):
	"""Test creating a temporary directory."""
	input_pdb_file, _ = common_test_data
	protein = Protein(str(input_pdb_file))
	temp_dir = protein.create_temp_dir()
	assert temp_dir.exists()
	assert temp_dir.is_dir()
	Protein.remove_temp_dir(temp_dir)
	assert not temp_dir.exists()


# Add more tests as needed for any additional functionality in the Protein class
