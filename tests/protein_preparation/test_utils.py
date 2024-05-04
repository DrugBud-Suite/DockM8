import pytest
import sys
from pathlib import Path

# Search for 'DockM8' in parent directories
dockm8_path = next((p / 'DockM8' for p in Path(__file__).resolve().parents if (p / 'DockM8').is_dir()), None)
sys.path.append(str(dockm8_path))

from scripts.protein_preparation.utils import extract_chain
from Bio.PDB import PDBParser

@pytest.fixture
def pdb_file():
    return dockm8_path / 'tests/test_files/1fvv_p.pdb'

def test_extract_chain_valid_chain(pdb_file):
    chain_id = 'A'
    output_file = extract_chain(pdb_file, chain_id)
    assert output_file.exists()
    assert output_file.name == f"{pdb_file.stem}_{chain_id}.pdb"
    parser = PDBParser()
    structure = parser.get_structure('structure', output_file)
    assert structure is not None