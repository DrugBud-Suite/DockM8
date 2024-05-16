import os
import sys
from pathlib import Path

import pytest
from Bio.PDB import PDBParser

# Search for 'DockM8' in parent directories
tests_path = next((p / 'tests' for p in Path(__file__).resolve().parents if (p / 'tests').is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.protonation.protonate_protoss import (
    protonate_protein_protoss,
)


@pytest.fixture
def common_test_data():
    """Set up common test data."""
    dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
    input_pdb_file = dockm8_path / "tests/test_files/1fvv_p.pdb"
    output_dir = dockm8_path / "tests/test_files"
    return input_pdb_file, output_dir


def test_protonate_protein_protoss(common_test_data):
    """Test protein preparation with ProtoSS."""
    input_pdb_file, output_dir = common_test_data
    output_path = protonate_protein_protoss(input_pdb_file, output_dir)
    # Check if the prepared protein file exists
    assert output_path.is_file()
    # Verify that the prepared protein file has the correct extension
    assert (
        output_path.name
        == f"{input_pdb_file.stem}_protoss{input_pdb_file.suffix}"
    )
    # Verify that the prepared protein file is not empty
    assert output_path.stat().st_size > 0
    # Verify that the prepared protein file is in the correct directory
    assert output_path.parent == output_dir
    # Check if the result is a readable PDB file
    parser = PDBParser()
    structure = parser.get_structure("protein", str(output_path))
    assert structure is not None
    os.unlink(output_path) if os.path.exists(output_path) else None

