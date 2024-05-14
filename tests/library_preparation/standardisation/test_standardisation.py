import os
import sys
from pathlib import Path

import pytest
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
dockm8_path = next((p / 'DockM8' for p in Path(__file__).resolve().parents if (p / 'DockM8').is_dir()), None)
sys.path.append(str(dockm8_path))

from scripts.library_preparation.standardisation.standardise import standardize_library

@pytest.fixture
def common_test_data():
    """Set up common test data."""
    library = dockm8_path / "tests/test_files/library_preparation/library.sdf"
    output_dir = dockm8_path / "tests/test_files/library_preparation/"
    id_column = "ID"
    return library, output_dir, id_column

@pytest.fixture
def cleanup(request):
    """Cleanup fixture to remove generated files after each test."""
    output_dir = dockm8_path / "tests/test_files/library_preparation/"

    def remove_created_files():
        for file in output_dir.iterdir():
            if file.name in ["standardized_library.sdf"]:
                file.unlink()

    request.addfinalizer(remove_created_files)

def test_standardize_library(common_test_data, cleanup):
    # Define the input parameters for the function
    library, output_dir, id_column = common_test_data
    n_cpus = int(os.cpu_count() * 0.9)
    # Call the function
    standardized_file = standardize_library(library, output_dir, id_column, n_cpus)
    # Add your assertions here to verify the expected behavior of the function

    # Verify that the standardized library file exists
    assert (output_dir / "standardized_library.sdf").exists()

    # Verify that the standardized library file is not empty
    assert (output_dir / "standardized_library.sdf").stat().st_size > 0

    # Verify that the standardized library file is a valid SDF file
    try:
        PandasTools.LoadSDF(str(output_dir / "standardized_library.sdf"))
    except Exception as e:
        assert False, f"Failed to load standardized library SDF file: {str(e)}"

    # Verify that the number of compounds in the standardized library is the same as the input library
    input_df = PandasTools.LoadSDF(str(library), molColName=None, idName=id_column)
    standardized_df = PandasTools.LoadSDF(str(standardized_file), molColName=None, idName=id_column)
    assert len(input_df) == len(standardized_df)

    # Verify that the standardized library does not contain any duplicate compounds
    assert len(standardized_df.drop_duplicates(subset="ID")) == len(standardized_df)