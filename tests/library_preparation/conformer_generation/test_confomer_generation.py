import os
import sys
from pathlib import Path
import pytest
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests"
                   for p in Path(__file__).resolve().parents
                   if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.library_preparation.conformer_generation.confgen_GypsumDL import generate_conformers_GypsumDL
from scripts.library_preparation.conformer_generation.confgen_RDKit import generate_conformers_RDKit


@pytest.fixture
def common_test_data():
    """Set up common test data."""
    dockm8_path = next((p / "tests"
                        for p in Path(__file__).resolve().parents
                        if (p / "tests").is_dir()), None).parent
    library = dockm8_path / "tests/test_files/library_preparation/library.sdf"
    output_dir = dockm8_path / "tests/test_files/library_preparation/"
    software = dockm8_path / "software"
    return library, output_dir, software


@pytest.fixture
def cleanup(request):
    """Cleanup fixture to remove generated files after each test."""
    output_dir = dockm8_path / "tests/test_files/library_preparation/"

    def remove_created_files():
        for file in output_dir.iterdir():
            if file.name in [
                    "final_library.sdf", "protonated_library.sdf",
                    "standardized_library.sdf"]:
                file.unlink()

    request.addfinalizer(remove_created_files)


def test_generate_conformers_GypsumDL(common_test_data, cleanup):
    """Test generate_conformers_GypsumDL function."""
    library, output_dir, software = common_test_data
    n_cpus = int(os.cpu_count() * 0.9)

    library_df = PandasTools.LoadSDF(str(library), molColName=None, idName="ID")

    # Call the function
    output_file = generate_conformers_GypsumDL(library, output_dir, software,
                                               n_cpus)

    output_df = PandasTools.LoadSDF(str(output_file),
                                    molColName=None,
                                    idName="ID")

    # Check if the output file exists
    assert output_file.exists()

    # Check if the output file is in the correct directory
    assert output_file.name == "generated_conformers.sdf"

    # Check if the number of molecules in the input and output files are the same
    assert len(library_df) == len(output_df)

    # Check if the unnecessary files are correctly removed
    assert not (output_dir / "GypsumDL_results").exists()
    assert not (output_dir / "GypsumDL_split").exists()
    assert not (output_dir / "gypsum_dl_success.sdf").exists()
    assert not (output_dir / "gypsum_dl_failed.smi").exists()
    os.remove(output_file) if output_file.exists() else None


def test_generate_conformers_RDKit(common_test_data, cleanup):
    """Test generate_conformers_RDKit function."""
    library, output_dir, software = common_test_data
    n_cpus = int(os.cpu_count() * 0.9)

    library_df = PandasTools.LoadSDF(str(library), molColName=None, idName="ID")

    # Call the function
    output_file = generate_conformers_RDKit(library, output_dir, n_cpus)

    output_df = PandasTools.LoadSDF(str(output_file),
                                    molColName=None,
                                    idName="ID")

    # Check if the output file exists
    assert output_file.exists()

    # Check if the output file is in the correct directory
    assert output_file.parent == output_dir

    # Check if the output file has the correct name
    assert output_file.name == "generated_conformers.sdf"

    # Check if the number of molecules in the input and output files are the same
    assert len(library_df) == len(output_df)
    os.remove(output_file) if output_file.exists() else None
