import os
from pathlib import Path
import sys

import pandas as pd

import pytest


# Search for 'DockM8' in parent directories
tests_path = next(
    (p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()),
    None,
)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring import rescore_poses
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS
from scripts.utilities.utilities import delete_files

@pytest.fixture
def test_data():
    dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
    w_dir = dockm8_path / "tests/test_files/rescoring"
    protein_file = dockm8_path / "tests/test_files/rescoring/example_prepared_receptor_1fvv.pdb"
    software = dockm8_path / "software"
    clustered_sdf = dockm8_path / "tests/test_files/rescoring/example_poses_1fvv.sdf"
    functions = [key for key in RESCORING_FUNCTIONS.keys() if key not in ["AAScore", "PLECScore", "RFScoreVS"]]
    n_cpus = int(os.cpu_count() * 0.9)
    return w_dir, protein_file, software, clustered_sdf, functions, n_cpus

def test_rescore_poses(test_data):
    w_dir, protein_file, software, clustered_sdf, functions, n_cpus = test_data
    # Ensure that output directory is clean
    rescoring_folder = w_dir / f"rescoring_{clustered_sdf.stem}"
    if rescoring_folder.exists():
        delete_files(rescoring_folder, None)

    rescore_poses(w_dir, protein_file, software, clustered_sdf, functions, n_cpus)

    # Check the presence of output files
    for function in functions:
        score_file = rescoring_folder / f"{function}_rescoring" / f"{function}_scores.csv"
        assert score_file.is_file(), f"Missing file: {score_file}"

    allposes_rescored_file = rescoring_folder / "allposes_rescored.csv"
    assert allposes_rescored_file.is_file(), f"Missing file: {allposes_rescored_file}"

    # Verify the content of the output files
    combined_scores = pd.read_csv(allposes_rescored_file)
    assert not combined_scores.empty, "All poses rescored output is empty"

    # Check if columns in the combined CSV match the expected format
    expected_columns = ["Pose ID"] + [RESCORING_FUNCTIONS[func]['column_name'] for func in functions]
    assert all(column in combined_scores.columns for column in expected_columns), "Missing expected columns in all poses rescored output"