import pytest
from pathlib import Path
import os
import sys

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.protein_preparation import prepare_protein
from Bio.PDB import PDBParser


@pytest.fixture
def common_test_data():
    """Set up common test data."""
    dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
    input_pdb_file = dockm8_path / "test_data/1fvv_p.pdb"
    output_dir = dockm8_path / "test_data/protein_preparation/"
    return input_pdb_file, output_dir


@pytest.fixture
def cleanup(request):
    """Cleanup fixture to remove generated files after each test."""
    output_dir = dockm8_path / "test_data/protein_preparation/"

    def remove_created_files():
        for file in output_dir.iterdir():
            if file.name in ["prepared_receptor.pdb", "2O1X.pdb"]:
                file.unlink()

    request.addfinalizer(remove_created_files)


def test_protoss_protonation(common_test_data, cleanup):
    """Test using Protoss for protonation."""
    protein_file_or_code, output_dir = common_test_data
    output_path = prepare_protein(protein_file_or_code, output_dir=output_dir, protonation_method="protoss")
    assert isinstance(output_path, Path)
    assert output_path.exists()
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None


def test_pdbfixer_default_protonation(common_test_data, cleanup):
    """Test using PDBFixer with default pH."""
    protein_file_or_code, output_dir = common_test_data
    output_path = prepare_protein(protein_file_or_code, output_dir=output_dir, protonation_method="pdbfixer")
    assert isinstance(output_path, Path)
    assert output_path.exists()
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None


def test_pdbfixer_custom_ph(common_test_data, cleanup):
    """Test using PDBFixer with custom pH."""
    protein_file_or_code, output_dir = common_test_data
    output_path = prepare_protein(protein_file_or_code, output_dir=output_dir, protonation_method=7.4)
    assert isinstance(output_path, Path)
    assert output_path.exists()
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None


def test_prepare_protein_with_file_input(common_test_data, cleanup):
    """Test case for preparing protein with file input."""
    input_pdb_file, output_dir = common_test_data
    output_path = prepare_protein(
        protein_file_or_code=input_pdb_file,
        output_dir=output_dir,
        fix_nonstandard_residues=True,
        fix_missing_residues=True,
        protonation_method="protoss",
        remove_hetero=True,
        remove_water=True,
    )
    assert isinstance(output_path, Path)
    assert output_path.exists()
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None


def test_prepare_protein_with_pdb_input(common_test_data, cleanup):
    """Test case for preparing a protein with PDB input."""
    _, output_dir = common_test_data
    output_path = prepare_protein(
        "2o1x",
        output_dir=output_dir,
        fix_nonstandard_residues=True,
        fix_missing_residues=True,
        protonation_method="protoss",
        remove_hetero=True,
        remove_water=True,
    )
    assert isinstance(output_path, Path)
    assert output_path.exists()
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None


def test_prepare_protein_with_uniprot_input(common_test_data, cleanup):
    """Test case for preparing a protein with Uniprot input."""
    _, output_dir = common_test_data
    output_path = prepare_protein(
        "Q221Q3",
        output_dir=output_dir,
        fix_nonstandard_residues=True,
        fix_missing_residues=True,
        protonation_method="protoss",
        remove_hetero=True,
        remove_water=True,
    )
    assert isinstance(output_path, Path)
    assert output_path.exists()
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None


def test_prepare_protein_with_invalid_input(common_test_data, cleanup):
    """Test case to verify behavior with invalid input."""
    _, output_dir = common_test_data

    # Test invalid PDB code
    with pytest.raises(ValueError) as exc_info:
        prepare_protein("abcd", output_dir=Path(output_dir))
    assert "Invalid PDB code" in str(exc_info.value)

    # Test invalid Uniprot code
    with pytest.raises(ValueError) as exc_info:
        prepare_protein("abcdef", output_dir=Path(output_dir))
    assert "Failed to fetch AlphaFold structure" in str(exc_info.value)

    # Test invalid file path
    with pytest.raises(ValueError) as exc_info:
        prepare_protein("/invalid_input", output_dir=Path(output_dir))
    assert "Invalid input" in str(exc_info.value)

    # Test invalid input length
    with pytest.raises(ValueError) as exc_info:
        prepare_protein("a", output_dir=Path(output_dir))
    assert "Invalid input" in str(exc_info.value)

    # Test invalid protonation method
    with pytest.raises(ValueError) as exc_info:
        prepare_protein("2o1x", output_dir=Path(output_dir), protonation_method="invalid")
    assert "Protonation method must be" in str(exc_info.value)

    # Test invalid pH value
    with pytest.raises(ValueError) as exc_info:
        prepare_protein("2o1x", output_dir=Path(output_dir), protonation_method=15.0)
    assert "pH value must be between" in str(exc_info.value)


def test_prepare_protein_without_modifications(common_test_data, cleanup):
    """Test preparing protein without any modifications."""
    protein_file_or_code, output_dir = common_test_data
    output_path = prepare_protein(
        protein_file_or_code,
        output_dir=output_dir,
        fix_nonstandard_residues=False,
        fix_missing_residues=False,
        remove_hetero=False,
        remove_water=False,
        protonation_method="protoss",
    )
    assert isinstance(output_path, Path)
    assert output_path.exists()
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None


def test_prepare_protein_partial_modifications(common_test_data, cleanup):
    """Test preparing protein with only some modifications enabled."""
    protein_file_or_code, output_dir = common_test_data
    output_path = prepare_protein(
        protein_file_or_code,
        output_dir=output_dir,
        fix_nonstandard_residues=True,
        fix_missing_residues=False,
        remove_hetero=True,
        remove_water=False,
        protonation_method="pdbfixer",
    )
    assert isinstance(output_path, Path)
    assert output_path.exists()
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
