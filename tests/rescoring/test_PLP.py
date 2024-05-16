import os
from pathlib import Path
import sys

from pandas import DataFrame

import pytest

# Search for 'DockM8' in parent directories
tests_path = next(
    (p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()),
    None,
)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring_functions.PLP import plp_rescoring


@pytest.fixture
def test_data():
    dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
    w_dir = dockm8_path / "tests/test_files/rescoring"
    protein_file = dockm8_path / "tests/test_files/rescoring/example_prepared_receptor_1fvv.pdb"
    software = dockm8_path / "software"
    clustered_sdf = dockm8_path / "tests/test_files/rescoring/example_poses_1fvv.sdf"
    n_cpus = int(os.cpu_count() * 0.9)
    pocket_definition = {"center": [1.0, 2.0, 3.0], "size": [10.0]}
    return w_dir, protein_file, software, clustered_sdf, n_cpus, pocket_definition


def test_plp_rescoring(test_data):
    # Define the input arguments for the function
    w_dir, protein_file, software, clustered_sdf, n_cpus, pocket_definition = test_data
    column_name = "PLP"
    rescoring_folder = w_dir / f"rescoring_{clustered_sdf.stem}"

    # Call the function
    result = plp_rescoring(clustered_sdf, 
                           n_cpus, 
                           column_name, 
                           rescoring_folder=rescoring_folder, 
                           software=software, 
                           protein_file=protein_file, 
                           pocket_definition=pocket_definition)

    # Assert the result
    assert isinstance(result, DataFrame)
    assert "Pose ID" in result.columns
    assert "PLP" in result.columns
    assert len(result) > 0
