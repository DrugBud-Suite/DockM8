import pytest
import sys
from pathlib import Path

# Search for 'DockM8' in parent directories
dockm8_path = next((p / 'DockM8' for p in Path(__file__).resolve().parents if (p / 'DockM8').is_dir()), None)
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.fetching.fetch_pdb import fetch_pdb_structure
from scripts.protein_preparation.fetching.fetch_alphafold import fetch_alphafold_structure

@pytest.fixture
def common_test_data():
    valid_pdb_id = "1obv"
    invalid_pdb_id = "invalid_id"
    valid_uniprot_code = "P00520"
    invalid_uniprot_code = "invalid_code"
    output_dir = dockm8_path / "tests/test_files"
    """Set up the output directory."""
    return valid_pdb_id, invalid_pdb_id, valid_uniprot_code, invalid_uniprot_code, output_dir

def test_fetch_pdb_structure_success(common_test_data):
    valid_pdb_id, invalid_pdb_id, valid_uniprot_code, invalid_uniprot_code, output_dir = common_test_data
    output_path = fetch_pdb_structure(valid_pdb_id, output_dir)
    assert output_path.exists()
    assert output_path.name == f"{valid_pdb_id}.pdb"

def test_fetch_pdb_structure_failure(common_test_data):
    valid_pdb_id, invalid_pdb_id, valid_uniprot_code, invalid_uniprot_code, output_dir = common_test_data
    with pytest.raises(Exception):
        output_path = fetch_pdb_structure(invalid_pdb_id, output_dir)
        assert not output_path.exists()
    
def test_fetch_alphafold_structure_success(common_test_data):
    valid_pdb_id, invalid_pdb_id, valid_uniprot_code, invalid_uniprot_code, output_dir = common_test_data
    output_path = fetch_alphafold_structure(valid_uniprot_code, output_dir)
    assert output_path.exists()
    assert output_path.name == f"{valid_uniprot_code}.pdb"

def test_fetch_alphafold_structure_failure(common_test_data):
    valid_pdb_id, invalid_pdb_id, valid_uniprot_code, invalid_uniprot_code, output_dir = common_test_data
    with pytest.raises(Exception):
        output_path = fetch_alphafold_structure(invalid_uniprot_code, output_dir)
        assert not output_path.exists()