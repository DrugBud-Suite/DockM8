import os
import sys
from pathlib import Path

import pytest
from Bio.PDB import PDBParser

# Search for 'DockM8' in parent directories
tests_path = next((p / 'tests' for p in Path(__file__).resolve().parents if (p / 'tests').is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.fixing.pdb_fixer import fix_pdb_file


@pytest.fixture
def common_test_data():
    """Set up common test data."""
    input_pdb_file = dockm8_path / "tests/test_files/1fvv_p.pdb"
    output_dir = dockm8_path / "tests/test_files"
    return input_pdb_file, output_dir


def test_fix_pdb_file_default(common_test_data):
    """Test fixing a PDB file with default options."""
    input_pdb_file, output_dir = common_test_data
    output_path = fix_pdb_file(input_pdb_file, output_dir)
    assert output_path.is_file()
    assert output_path.name == f"{input_pdb_file.stem}_fixed{input_pdb_file.suffix}"
    # Check if the result is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None



def test_fix_pdb_file_custom_options(common_test_data):
    """Test fixing a PDB file with custom options."""
    input_pdb_file, output_dir = common_test_data
    output_path = fix_pdb_file(
        input_pdb_file,
        output_dir,
        fix_nonstandard_residues=False,
        fix_missing_residues=False,
        add_missing_hydrogens_pH=7.4,
        remove_hetero=False,
        remove_water=True,
    )
    assert output_path.is_file()
    assert output_path.name == f"{input_pdb_file.stem}_fixed{input_pdb_file.suffix}"
    assert output_path.parent == output_dir
    # Check if the result is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None



def test_fix_pdb_file_no_output_dir(common_test_data):
    """Test fixing a PDB file without specifying an output directory."""
    input_pdb_file, output_dir = common_test_data
    output_path = fix_pdb_file(input_pdb_file, None)
    assert output_path.is_file()
    assert output_path.name == f"{input_pdb_file.stem}_fixed{input_pdb_file.suffix}"
    assert output_path.parent == input_pdb_file.parent
    # Check if the result is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None

