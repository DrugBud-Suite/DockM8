import sys
import os
from pathlib import Path

import pytest

# Search for 'DockM8' in parent directories
tests_path = next((p / 'tests' for p in Path(__file__).resolve().parents if (p / 'tests').is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from Bio.PDB import PDBParser

from scripts.protein_preparation.fetching.fetch_alphafold import (
    fetch_alphafold_structure,
)
from scripts.protein_preparation.fetching.fetch_pdb import fetch_pdb_structure


@pytest.fixture
def common_test_data():
    valid_pdb_id = "1obv"
    invalid_pdb_id = "invalid_id"
    valid_uniprot_code = "P00520"
    invalid_uniprot_code = "invalid_code"
    output_dir = dockm8_path / "tests/test_files"
    """Set up the output directory."""
    return (
        valid_pdb_id,
        invalid_pdb_id,
        valid_uniprot_code,
        invalid_uniprot_code,
        output_dir,
    )


def test_fetch_pdb_structure_success(common_test_data):
    """
    Test case to verify the successful fetching of a PDB structure.

    Args:
        common_test_data: A tuple containing the common test data.

    Returns:
        None

    Raises:
        AssertionError: If any of the assertions fail.
    """
    (
        valid_pdb_id,
        invalid_pdb_id,
        valid_uniprot_code,
        invalid_uniprot_code,
        output_dir,
    ) = common_test_data
    output_path = fetch_pdb_structure(valid_pdb_id, output_dir)
    assert output_path.exists()
    assert output_path.name == f"{valid_pdb_id}.pdb"
    # Check if the result is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None



def test_fetch_pdb_structure_failure(common_test_data):
    """
    Test case to verify the failure scenario of fetching PDB structure.

    Args:
        common_test_data: A tuple containing the common test data.

    Raises:
        Exception: If an exception is not raised during the execution of the test.

    Returns:
        None
    """
    (
        valid_pdb_id,
        invalid_pdb_id,
        valid_uniprot_code,
        invalid_uniprot_code,
        output_dir,
    ) = common_test_data
    with pytest.raises(Exception):
        output_path = fetch_pdb_structure(invalid_pdb_id, output_dir)
        assert not output_path.exists()


def test_fetch_alphafold_structure_success(common_test_data):
    """
    Test case for fetching the AlphaFold structure successfully.

    Args:
        common_test_data: A tuple containing the common test data.

    Returns:
        None

    Raises:
        AssertionError: If any of the assertions fail.
    """
    (
        valid_pdb_id,
        invalid_pdb_id,
        valid_uniprot_code,
        invalid_uniprot_code,
        output_dir,
    ) = common_test_data
    output_path = fetch_alphafold_structure(valid_uniprot_code, output_dir)
    assert output_path.exists()
    assert output_path.name == f"{valid_uniprot_code}.pdb"
    # Check if the result is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None



def test_fetch_alphafold_structure_failure(common_test_data):
    """
    Test case to verify the failure scenario of fetching an AlphaFold structure.

    Args:
        common_test_data: A tuple containing common test data.

    Raises:
        Exception: If fetching the AlphaFold structure for an invalid UniProt code does not raise an exception.

    Returns:
        None
    """
    (
        valid_pdb_id,
        invalid_pdb_id,
        valid_uniprot_code,
        invalid_uniprot_code,
        output_dir,
    ) = common_test_data
    with pytest.raises(Exception):
        output_path = fetch_alphafold_structure(invalid_uniprot_code, output_dir)
        assert not output_path.exists()
