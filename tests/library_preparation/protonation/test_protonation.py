import os
import sys
from pathlib import Path

import pytest
from rdkit.Chem import PandasTools

cwd = Path.cwd()
dockm8_path = next((path for path in cwd.parents if path.name == "DockM8"), None)
sys.path.append(str(dockm8_path))

from scripts.library_preparation.protonation.protgen_GypsumDL import protonate_GypsumDL

@pytest.fixture
def common_test_data():
    """Set up common test data."""
    library = dockm8_path / 'tests/test_files/library.sdf'
    output_dir = dockm8_path / 'tests/test_files/'
    software = dockm8_path / 'software'
    return library, output_dir, software

def test_protonate_GypsumDL(common_test_data):
    library, output_dir, software = common_test_data
    ncpus = int(os.cpu_count()*0.9)
    
    library_df = PandasTools.LoadSDF(str(library), molColName=None, idName='ID')
    
    output_file = protonate_GypsumDL(library, output_dir, software, ncpus)
    
    output_df = PandasTools.LoadSDF(str(output_file), molColName=None, idName='ID')
    
    assert output_file.exists()
    assert output_file.name == "protonated_library.sdf"
    assert len(library_df) == len(output_df)
    assert not (output_dir / "GypsumDL_results").exists()
    assert not (output_dir / "GypsumDL_split").exists()
    assert not (output_dir / "gypsum_dl_success.sdf").exists()
    assert not (output_dir / "gypsum_dl_failed.smi").exists()
