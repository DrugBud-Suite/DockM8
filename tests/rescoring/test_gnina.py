import os
from pathlib import Path
import sys

from pandas import DataFrame

import pytest

# Search for 'DockM8' in parent directories
tests_path = next(
    (p / "tests"
     for p in Path(__file__).resolve().parents
     if (p / "tests").is_dir()),
    None,
)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring_functions.gnina import gnina_rescoring


@pytest.fixture
def test_data():
    dockm8_path = next(
        (p / "tests"
         for p in Path(__file__).resolve().parents
         if (p / "tests").is_dir()),
        None,
    ).parent
    w_dir = dockm8_path / "tests/test_files/rescoring"
    protein_file = (
        dockm8_path /
        "tests/test_files/rescoring/example_prepared_receptor_1fvv.pdb")
    software = dockm8_path / "software"
    clustered_sdf = dockm8_path / "tests/test_files/rescoring/example_poses_1fvv.sdf"
    n_cpus = int(os.cpu_count() * 0.9)
    return w_dir, protein_file, software, clustered_sdf, n_cpus


def test_gnina_rescoring_affinity(test_data):
    # Define the input arguments for the function
    w_dir, protein_file, software, clustered_sdf, n_cpus = test_data
    column_name = "GNINA-Affinity"
    rescoring_folder = w_dir / f"rescoring_{clustered_sdf.stem}"

    # Call the function
    result = gnina_rescoring(
        clustered_sdf,
        n_cpus,
        column_name,
        rescoring_folder=rescoring_folder,
        software=software,
        protein_file=protein_file,
    )

    # Assert the result
    assert isinstance(result, DataFrame)
    assert "Pose ID" in result.columns
    assert "GNINA-Affinity" in result.columns
    assert len(result) > 0


def test_gnina_rescoring_cnnscore(test_data):
    # Define the input arguments for the function
    w_dir, protein_file, software, clustered_sdf, n_cpus = test_data
    column_name = "CNN-Score"
    rescoring_folder = w_dir / f"rescoring_{clustered_sdf.stem}"

    # Call the function
    result = gnina_rescoring(
        clustered_sdf,
        n_cpus,
        column_name,
        rescoring_folder=rescoring_folder,
        software=software,
        protein_file=protein_file,
    )

    # Assert the result
    assert isinstance(result, DataFrame)
    assert "Pose ID" in result.columns
    assert "CNN-Score" in result.columns
    assert len(result) > 0


def test_gnina_rescoring_cnnaffinity(test_data):
    # Define the input arguments for the function
    w_dir, protein_file, software, clustered_sdf, n_cpus = test_data
    column_name = "CNN-Affinity"
    rescoring_folder = w_dir / f"rescoring_{clustered_sdf.stem}"

    # Call the function
    result = gnina_rescoring(
        clustered_sdf,
        n_cpus,
        column_name,
        rescoring_folder=rescoring_folder,
        software=software,
        protein_file=protein_file,
    )

    # Assert the result
    assert isinstance(result, DataFrame)
    assert "Pose ID" in result.columns
    assert "CNN-Affinity" in result.columns
    assert len(result) > 0