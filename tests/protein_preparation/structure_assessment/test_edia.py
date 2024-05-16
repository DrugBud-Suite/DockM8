import os
import sys
from pathlib import Path

import pytest
from Bio.PDB import PDBParser

# Search for 'DockM8' in parent directories
tests_path = next((p / 'tests'
                   for p in Path(__file__).resolve().parents
                   if (p / 'tests').is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.structure_assessment.edia import get_best_chain_edia


@pytest.fixture
def common_test_data():
    """Set up common test data."""
    pdb_code = "2o1x"
    output_dir = dockm8_path / "tests/test_files"
    return pdb_code, output_dir


def test_get_best_chain_edia(common_test_data):
    """Test getting the best chain using EDIA."""
    pdb_code, output_dir = common_test_data
    output_path = get_best_chain_edia(pdb_code, output_dir)
    expected_chain_path = output_dir / f"{pdb_code}_A.pdb"
    fetched_file = output_dir / f"{pdb_code}.pdb"
    assert output_path.is_file()
    assert output_path == expected_chain_path
    # Verify that the extracted chain file is not empty
    assert output_path.stat().st_size > 0
    parser = PDBParser()
    structure = parser.get_structure("structure", output_path)
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None
    os.unlink(fetched_file) if os.path.exists(fetched_file) else None


def test_get_best_chain_edia_invalid_pdb_code(common_test_data):
    """Test getting the best chain using EDIA with an invalid PDB code."""
    pdb_code, output_dir = common_test_data
    pdb_code = "invalid"
    with pytest.raises(Exception):
        get_best_chain_edia(pdb_code, output_dir)
