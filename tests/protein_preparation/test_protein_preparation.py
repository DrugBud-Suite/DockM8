import pytest
from pathlib import Path
import os
import sys

# Search for 'DockM8' in parent directories
tests_path = next((p / 'tests'
                   for p in Path(__file__).resolve().parents
                   if (p / 'tests').is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.protein_preparation import prepare_protein
from Bio.PDB import PDBParser


@pytest.fixture
def common_test_data():
    """Set up common test data."""
    dockm8_path = next((p / "tests"
                        for p in Path(__file__).resolve().parents
                        if (p / "tests").is_dir()), None).parent
    input_pdb_file = dockm8_path / 'tests/test_files/protein_preparation/1fvv_p.pdb'
    output_dir = dockm8_path / 'tests/test_files/protein_preparation'
    return input_pdb_file, output_dir


def test_prepare_protein_with_file_input(common_test_data):
    """
    Test case for preparing protein with file input.

    Args:
        common_test_data: A tuple containing the input file and output directory.

    Returns:
        None

    Raises:
        AssertionError: If the output_path is not an instance of Path or if the output_path does not exist.
    """
    protein_file_or_code, output_dir = common_test_data
    output_path = prepare_protein(protein_file_or_code, output_dir=output_dir)
    assert isinstance(output_path, Path)
    assert output_path.exists()
    # Check if the output_path is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None


def test_prepare_protein_with_pdb_input(common_test_data):
    """
    Test case for preparing a protein with PDB input.

    Args:
        common_test_data: Tuple containing input file and output directory.

    Returns:
        None
    """
    protein_file_or_code, output_dir = common_test_data
    os.remove(output_dir / "prepared_receptor.pdb") if os.path.exists(
        output_dir / "prepared_receptor.pdb") else None
    output_path = prepare_protein("2o1x", output_dir=output_dir)
    assert isinstance(output_path, Path)
    assert output_path.exists()
    # Check if the output_path is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None
    os.unlink(output_dir / "2O1X.pdb") if os.path.exists(output_dir /
                                                         "2O1X.pdb") else None


def test_prepare_protein_with_uniprot_input(common_test_data):
    """
    Test case for preparing a protein with Uniprot input.

    Args:
        common_test_data: Tuple containing the input file and output directory.

    Returns:
        None

    Raises:
        AssertionError: If the output_path is not an instance of Path or if the output_path does not exist.
    """
    protein_file_or_code, output_dir = common_test_data
    os.remove(output_dir / "prepared_receptor.pdb") if os.path.exists(
        output_dir / "prepared_receptor.pdb") else None
    output_path = prepare_protein("P00520", output_dir=output_dir)
    assert isinstance(output_path, Path)
    assert output_path.exists()
    # Check if the output_path is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None


def test_prepare_protein_with_invalid_input(common_test_data):
    """
    Test case to verify the behavior of the prepare_protein function when given invalid input.

    Args:
        common_test_data: A tuple containing the input file and output directory.

    Raises:
        Exception: If the prepare_protein function does not raise an exception when given invalid input.

    """
    protein_file_or_code, output_dir = common_test_data
    os.remove(output_dir / "prepared_receptor.pdb") if os.path.exists(
        output_dir / "prepared_receptor.pdb") else None
    # Test invalid PDB code
    with pytest.raises(Exception):
        prepare_protein("abcd", output_dir=Path(output_dir))

    # Test invalid Uniprot code
    with pytest.raises(Exception):
        prepare_protein("abcdef", output_dir=Path(output_dir))

    # Test invalid file path
    with pytest.raises(Exception):
        prepare_protein("/invalid_input", output_dir=Path(output_dir))

    # Test invalid input length
    with pytest.raises(Exception):
        prepare_protein("a", output_dir=Path(output_dir))

    with pytest.raises(Exception):
        prepare_protein("abcdefgh", output_dir=Path(output_dir))


def test_prepare_protein_with_select_best_chain(common_test_data):
    """
    Test case for preparing a protein with the option to select the best chain.

    Args:
        common_test_data (tuple): A tuple containing the input file and output directory.

    Returns:
        None
    """
    protein_file_or_code, output_dir = common_test_data
    os.remove(output_dir / "prepared_receptor.pdb") if os.path.exists(
        output_dir / "prepared_receptor.pdb") else None
    output_path = prepare_protein("2o1x",
                                  output_dir=output_dir,
                                  select_best_chain=True)
    assert isinstance(output_path, Path)
    assert output_path.exists()
    # Check if the output_path is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None
    os.unlink(output_dir / "2O1X.pdb") if os.path.exists(output_dir /
                                                         "2O1X.pdb") else None


def test_prepare_protein_without_fix_protein(common_test_data):
    """
    Test case for preparing protein without fixing the protein.

    Args:
        common_test_data: A tuple containing the input file and output directory.

    Returns:
        None

    Raises:
        AssertionError: If the output_path is not an instance of Path or if the output_path does not exist.
    """
    protein_file_or_code, output_dir = common_test_data
    os.remove(output_dir / "prepared_receptor.pdb") if os.path.exists(
        output_dir / "prepared_receptor.pdb") else None
    output_path = prepare_protein(protein_file_or_code,
                                  output_dir=output_dir,
                                  fix_protein=False)
    assert isinstance(output_path, Path)
    assert output_path.exists()
    # Check if the output_path is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None


def test_prepare_protein_without_fix_nonstandard_residues(common_test_data):
    """
    Test case for preparing protein without fixing nonstandard residues.

    Args:
        common_test_data: A tuple containing the input file and output directory.

    Returns:
        None
    """
    protein_file_or_code, output_dir = common_test_data
    os.remove(output_dir / "prepared_receptor.pdb") if os.path.exists(
        output_dir / "prepared_receptor.pdb") else None
    output_path = prepare_protein(protein_file_or_code,
                                  output_dir=output_dir,
                                  fix_nonstandard_residues=False)
    assert isinstance(output_path, Path)
    assert output_path.exists()
    # Check if the output_path is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None


def test_prepare_protein_without_fix_missing_residues(common_test_data):
    """
    Test case for preparing a protein without fixing missing residues.

    Args:
        common_test_data (tuple): A tuple containing the input file and output directory.

    Returns:
        None
    """
    protein_file_or_code, output_dir = common_test_data
    os.remove(output_dir / "prepared_receptor.pdb") if os.path.exists(
        output_dir / "prepared_receptor.pdb") else None
    output_path = prepare_protein(protein_file_or_code,
                                  output_dir=output_dir,
                                  fix_missing_residues=False)
    assert isinstance(output_path, Path)
    assert output_path.exists()
    # Check if the output_path is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None


def test_prepare_protein_without_add_missing_hydrogens_pH(common_test_data):
    """
    Test case for preparing protein without adding missing hydrogens at a specific pH.

    Args:
        common_test_data: A tuple containing the input file and output directory.

    Returns:
        None

    Raises:
        AssertionError: If the output_path is not an instance of Path or if the output_path does not exist.
    """
    protein_file_or_code, output_dir = common_test_data
    os.remove(output_dir / "prepared_receptor.pdb") if os.path.exists(
        output_dir / "prepared_receptor.pdb") else None
    output_path = prepare_protein(protein_file_or_code,
                                  output_dir=output_dir,
                                  add_missing_hydrogens_pH=None)
    assert isinstance(output_path, Path)
    assert output_path.exists()
    # Check if the output_path is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None


def test_prepare_protein_without_remove_hetero(common_test_data):
    """
    Test case for preparing protein without removing hetero atoms.

    Args:
        common_test_data: Tuple containing input file and output directory.

    Returns:
        None
    """
    protein_file_or_code, output_dir = common_test_data
    os.remove(output_dir / "prepared_receptor.pdb") if os.path.exists(
        output_dir / "prepared_receptor.pdb") else None
    output_path = prepare_protein(protein_file_or_code,
                                  output_dir=output_dir,
                                  remove_hetero=False)
    assert isinstance(output_path, Path)
    assert output_path.exists()
    # Check if the output_path is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None


def test_prepare_protein_without_remove_water(common_test_data):
    """
    Test case for preparing protein without removing water.

    Args:
        common_test_data: Tuple containing input file and output directory.

    Returns:
        None

    Raises:
        AssertionError: If the output_path is not an instance of Path or if the output_path does not exist.
    """
    protein_file_or_code, output_dir = common_test_data
    os.remove(output_dir / "prepared_receptor.pdb") if os.path.exists(
        output_dir / "prepared_receptor.pdb") else None
    output_path = prepare_protein(protein_file_or_code,
                                  output_dir=output_dir,
                                  remove_water=False)
    assert isinstance(output_path, Path)
    assert output_path.exists()
    # Check if the output_path is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None


def test_prepare_protein_without_protonate(common_test_data):
    """
    Test case for preparing protein without protonation.

    Args:
        common_test_data (tuple): A tuple containing the input file and output directory.

    Returns:
        None
    """
    protein_file_or_code, output_dir = common_test_data
    os.remove(output_dir / "prepared_receptor.pdb") if os.path.exists(
        output_dir / "prepared_receptor.pdb") else None
    output_path = prepare_protein(protein_file_or_code,
                                  output_dir=output_dir,
                                  protonate=False)
    assert isinstance(output_path, Path)
    assert output_path.exists()
    # Check if the output_path is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None
