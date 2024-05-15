import os
import sys
from pathlib import Path

import pytest
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
tests_path = next((p / 'tests' for p in Path(__file__).resolve().parents if (p / 'tests').is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.library_preparation.protonation.protgen_GypsumDL import protonate_GypsumDL


@pytest.fixture
def common_test_data():
    """Set up common test data."""
    library = dockm8_path / 'tests/test_files/library_preparation/library.sdf'
    output_dir = dockm8_path / 'tests/test_files/library_preparation/'
    software = dockm8_path / 'software'
    return library, output_dir, software

@pytest.fixture
def cleanup(request):
    """Cleanup fixture to remove generated files after each test."""
    output_dir = dockm8_path / "tests/test_files/library_preparation/"

    def remove_created_files():
        for file in output_dir.iterdir():
            if file.name in ["final_library.sdf", "protonated_library.sdf", "standardized_library.sdf"]:
                file.unlink()

    request.addfinalizer(remove_created_files)

def test_protonate_GypsumDL(common_test_data, cleanup):
    library, output_dir, software = common_test_data
    n_cpus = int(os.cpu_count()*0.9)
    
    library_df = PandasTools.LoadSDF(str(library), molColName=None, idName='ID')
    
    output_file = protonate_GypsumDL(library, output_dir, software, n_cpus)
    
    output_df = PandasTools.LoadSDF(str(output_file), molColName=None, idName='ID')
    
    assert output_file.exists()
    assert output_file.name == "protonated_library.sdf"
    assert len(library_df) == len(output_df)
    assert not (output_dir / "GypsumDL_results").exists()
    assert not (output_dir / "GypsumDL_split").exists()
    assert not (output_dir / "gypsum_dl_success.sdf").exists()
    assert not (output_dir / "gypsum_dl_failed.smi").exists()
