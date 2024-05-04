import sys

import pytest
from pathlib import Path

# Search for 'DockM8' in parent directories
dockm8_path = next((p / 'DockM8' for p in Path(__file__).resolve().parents if (p / 'DockM8').is_dir()), None)
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.fixing.pdb_fixer import fix_pdb_file

@pytest.fixture
def common_test_data():
    """Set up common test data."""
    input_pdb_file = dockm8_path / 'tests/test_files/1fvv_p.pdb'
    output_dir = dockm8_path / 'tests/test_files'
    return input_pdb_file, output_dir

def test_fix_pdb_file_default(common_test_data):
    """Test fixing a PDB file with default options."""
    input_pdb_file, output_dir = common_test_data
    fixed_pdb_file = fix_pdb_file(input_pdb_file, output_dir)
    assert fixed_pdb_file.is_file()
    assert fixed_pdb_file.name == f"{input_pdb_file.stem}_fixed{input_pdb_file.suffix}"

def test_fix_pdb_file_custom_options(common_test_data):
    """Test fixing a PDB file with custom options."""
    input_pdb_file, output_dir = common_test_data
    fixed_pdb_file = fix_pdb_file(
        input_pdb_file,
        output_dir,
        fix_nonstandard_residues=False,
        fix_missing_residues=False,
        add_missing_hydrogens_pH=7.4,
        remove_hetero=False,
        keep_water=False,
    )
    assert fixed_pdb_file.is_file()
    assert fixed_pdb_file.name == f"{input_pdb_file.stem}_fixed{input_pdb_file.suffix}"
    assert fixed_pdb_file.parent == output_dir

def test_fix_pdb_file_no_output_dir(common_test_data):
    """Test fixing a PDB file without specifying an output directory."""
    input_pdb_file, output_dir = common_test_data
    fixed_pdb_file = fix_pdb_file(input_pdb_file, None)
    assert fixed_pdb_file.is_file()
    assert fixed_pdb_file.name == f"{input_pdb_file.stem}_fixed{input_pdb_file.suffix}"
    assert fixed_pdb_file.parent == input_pdb_file.parent