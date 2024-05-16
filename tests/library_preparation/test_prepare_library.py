import os
import shutil
import sys
from pathlib import Path
import pytest
from rdkit.Chem import PandasTools
from scripts.library_preparation.main import prepare_library

# Search for 'DockM8' in parent directories
tests_path = next((p / 'tests' for p in Path(__file__).resolve().parents if (p / 'tests').is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))


@pytest.fixture
def common_test_data():
    """Set up common test data."""
    dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
    library = dockm8_path / "tests/test_files/library_preparation/library.sdf"
    output_dir = dockm8_path / "tests/test_files/library_preparation/"
    id_column = "ID"
    software = dockm8_path / "software"
    n_cpus = int(os.cpu_count() * 0.9)
    return library, output_dir, id_column, software, n_cpus

@pytest.fixture
def cleanup(request):
    """Cleanup fixture to remove generated files after each test."""
    output_dir = dockm8_path / "tests/test_files/library_preparation/"

    def remove_created_files():
        for file in output_dir.iterdir():
            if file.name in ["final_library.sdf", "protonated_library.sdf", "standardized_library.sdf"]:
                file.unlink()

    request.addfinalizer(remove_created_files)

def test_prepare_library_protonation(common_test_data, cleanup):
    """Test library preparation with standardization."""
    library, output_dir, id_column, software, n_cpus = common_test_data
    final_library = prepare_library(library, output_dir, id_column, "GypsumDL", "GypsumDL", software, n_cpus)
    expected_output = output_dir / "final_library.sdf"
    # Check if the final library file exists
    assert final_library == expected_output
    assert expected_output.is_file()
    # Verify that the number of compounds in the final library is the same as the input library
    input_df = PandasTools.LoadSDF(str(library), molColName=None, idName=id_column)
    final_library = PandasTools.LoadSDF(str(final_library), molColName=None, idName=id_column)
    assert len(input_df) == len(final_library)

def test_prepare_library_no_protonation(common_test_data, cleanup):
    """Test library preparation without protonation."""
    library, output_dir, id_column, software, n_cpus = common_test_data
    final_library = prepare_library(library, output_dir, id_column, "None", "GypsumDL", software, n_cpus)
    expected_output = output_dir / "final_library.sdf"
    # Check if the final library file exists
    assert final_library == expected_output
    assert expected_output.is_file()
    # Verify that the number of compounds in the final library is the same as the input library
    input_df = PandasTools.LoadSDF(str(library), molColName=None, idName=id_column)
    final_library = PandasTools.LoadSDF(str(final_library), molColName=None, idName=id_column)
    assert len(input_df) == len(final_library)

def test_prepare_library_invalid_protonation(common_test_data, cleanup):
    """Test library preparation with invalid protonation method."""
    library, output_dir, id_column, software, n_cpus = common_test_data
    # Check if the function raises a ValueError for an invalid protonation method
    with pytest.raises(ValueError):
        prepare_library(library, output_dir, id_column, "InvalidMethod", "GypsumDL", software, n_cpus)

def test_prepare_library_invalid_conformers(common_test_data, cleanup):
    """Test library preparation with invalid conformers method."""
    library, output_dir, id_column, software, n_cpus = common_test_data
    # Check if the function raises a ValueError for an invalid conformers method
    with pytest.raises(ValueError):
        prepare_library(library, output_dir, id_column, "GypsumDL", "InvalidMethod", software, n_cpus)
