import pytest
from pathlib import Path
import sys

# Search for 'DockM8' in parent directories
dockm8_path = next((p / 'DockM8' for p in Path(__file__).resolve().parents if (p / 'DockM8').is_dir()), None)
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.protonation.protonate_protoss import protonate_protein_protoss

@pytest.fixture
def common_test_data():
    """Set up common test data."""
    input_pdb_file = dockm8_path / 'tests/test_files/1fvv_p.pdb'
    output_dir = dockm8_path / 'tests/test_files'
    return input_pdb_file, output_dir

def test_protonate_protein_protoss(common_test_data):
    """Test protein preparation with ProtoSS."""
    input_pdb_file, output_dir = common_test_data
    protonated_protein = protonate_protein_protoss(input_pdb_file, output_dir)
    # Check if the prepared protein file exists
    assert protonated_protein.is_file()
    # Verify that the prepared protein file has the correct extension
    assert protonated_protein.name == f"{input_pdb_file.stem}_protoss{input_pdb_file.suffix}"
    # Verify that the prepared protein file is not empty
    assert protonated_protein.stat().st_size > 0
    # Verify that the prepared protein file is in the correct directory
    assert protonated_protein.parent == output_dir