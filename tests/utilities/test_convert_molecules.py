import os
import sys
from pathlib import Path
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.molecule_conversion import convert_molecules


@pytest.fixture
def cleanup(request):
	created_files = []

	def _cleanup():
		for file in created_files:
			if file.exists():
				file.unlink()

	def add_file(file):
		created_files.append(Path(file))

	request.addfinalizer(_cleanup)
	return add_file


@pytest.fixture
def common_test_data():
	"""Set up common test data."""
	dockm8_path = next((p for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
	test_files = dockm8_path / "tests/test_files/utilities/convert_molecules"
	software = dockm8_path / "software"
	return test_files, software


def is_valid_molecule(file_path, file_format):
	"""Check if the molecule in the file is valid using RDKit."""
	if file_format == "pdb":
		mol = Chem.MolFromPDBFile(str(file_path))
	elif file_format == "sdf":
		mol = Chem.SDMolSupplier(str(file_path))[0]
	else:
		raise ValueError(f"Unsupported file format: {file_format}")

	return mol is not None and mol.GetNumAtoms() > 0


def test_convert_pdb_to_pdbqt(common_test_data, cleanup, tmp_path):
	"""Test conversion from PDB to PDBQT format."""
	test_files, software = common_test_data
	input_file = test_files / "test_protein.pdb"
	output_file = tmp_path / "output_protein.pdbqt"

	result = convert_molecules(input_file, output_file, "pdb", "pdbqt")

	assert result is not None
	assert result.exists()
	assert result.suffix == ".pdbqt"

	cleanup(result)


def test_convert_sdf_to_pdbqt(common_test_data, cleanup, tmp_path):
	"""Test conversion from SDF to PDBQT format."""
	test_files, software = common_test_data
	input_file = test_files / "test_ligands.sdf"
	output_file = tmp_path / "output_ligands.pdbqt"

	result = convert_molecules(input_file, output_file, "sdf", "pdbqt")

	assert isinstance(result, list)
	assert all(file.suffix == ".pdbqt" for file in result)
	assert all(file.exists() for file in result)

	for file in result:
		cleanup(file)


def test_convert_sdf_to_mol2(common_test_data, cleanup, tmp_path):
	"""Test conversion from SDF to MOL2 format using Pybel."""
	test_files, software = common_test_data
	input_file = test_files / "test_ligands.sdf"
	output_file = tmp_path / "output_ligands.mol2"

	result = convert_molecules(input_file, output_file, "sdf", "mol2")

	assert result is not None
	assert result.exists()
	assert result.suffix == ".mol2"

	cleanup(result)


def test_convert_invalid_input(common_test_data):
	"""Test conversion with invalid input file."""
	test_files, software = common_test_data
	input_file = test_files / "non_existent_file.xyz"
	output_file = test_files / "output.pdbqt"

	with pytest.raises(FileNotFoundError):
		convert_molecules(input_file, output_file, "xyz", "pdbqt")


def test_convert_unsupported_format(common_test_data, tmp_path):
	"""Test conversion with unsupported output format."""
	test_files, software = common_test_data
	input_file = test_files / "test_ligands.sdf"
	output_file = tmp_path / "output.unsupported"

	with pytest.raises(ValueError):
		convert_molecules(input_file, output_file, "sdf", "unsupported")


# Helper function to create a test SDF file
def create_test_sdf(filepath):
	mol = Chem.MolFromSmiles("CCO")
	mol = Chem.AddHs(mol)
	AllChem.EmbedMolecule(mol)
	writer = Chem.SDWriter(str(filepath))
	writer.write(mol)
	writer.close()


def test_convert_single_molecule_sdf_to_pdbqt(common_test_data, cleanup, tmp_path):
	"""Test conversion of a single-molecule SDF to PDBQT format."""
	test_files, software = common_test_data
	input_file = tmp_path / "single_molecule.sdf"
	create_test_sdf(input_file)
	output_file = tmp_path / "output_single.pdbqt"

	result = convert_molecules(input_file, output_file, "sdf", "pdbqt")

	assert isinstance(result, list)
	assert len(result) == 1
	assert result[0].suffix == ".pdbqt"
	assert result[0].exists()

	cleanup(input_file)
	for file in result:
		cleanup(file)
