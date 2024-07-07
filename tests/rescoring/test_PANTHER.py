import os
from pathlib import Path
import sys

import pandas as pd
from pandas import DataFrame

import pytest

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring_functions.PANTHER import PANTHER_rescoring


@pytest.fixture
def test_data():
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	w_dir = dockm8_path / "tests/test_files/rescoring"
	software = dockm8_path / "software"
	protein_file = dockm8_path / "tests/test_files/rescoring/example_prepared_receptor_1fvv.pdb"
	clustered_sdf = dockm8_path / "tests/test_files/rescoring/example_poses_1fvv.sdf"
	pocket_definition = {"center": [-9.67, 207.73, 113.41], "size": [20.0, 20.0, 20.0]}
	n_cpus = int(os.cpu_count() * 0.9)
	return w_dir, software, protein_file, pocket_definition, clustered_sdf, n_cpus


def test_PANTHER_rescoring(test_data):
	# Define the input arguments for the function
	w_dir, software, protein_file, pocket_definition, clustered_sdf, n_cpus = test_data
	column_name = "PANTHER"
	rescoring_folder = w_dir / f"rescoring_{clustered_sdf.stem}"

	# Call the function
	result = PANTHER_rescoring(clustered_sdf,
			n_cpus,
			column_name,
			software=software,
			rescoring_folder=rescoring_folder,
			protein_file=protein_file,
			pocket_definition=pocket_definition)

	# Assert the result
	assert isinstance(result, DataFrame)
	assert "Pose ID" in result.columns
	assert "PANTHER" in result.columns
	assert len(result) > 0
	assert all(result["PANTHER"].notna())


def test_PANTHER_ESP_rescoring(test_data):
	# Define the input arguments for the function
	w_dir, software, protein_file, pocket_definition, clustered_sdf, n_cpus = test_data
	column_name = "PANTHER-ESP"
	rescoring_folder = w_dir / f"rescoring_{clustered_sdf.stem}"

	# Call the function
	result = PANTHER_rescoring(clustered_sdf,
			n_cpus,
			column_name,
			software=software,
			rescoring_folder=rescoring_folder,
			protein_file=protein_file,
			pocket_definition=pocket_definition)

	# Assert the result
	assert isinstance(result, DataFrame)
	assert "Pose ID" in result.columns
	assert "PANTHER-ESP" in result.columns
	assert len(result) > 0
	assert all(result["PANTHER-ESP"].notna())


def test_PANTHER_shape_rescoring(test_data):
	# Define the input arguments for the function
	w_dir, software, protein_file, pocket_definition, clustered_sdf, n_cpus = test_data
	column_name = "PANTHER-Shape"
	rescoring_folder = w_dir / f"rescoring_{clustered_sdf.stem}"

	# Call the function
	result = PANTHER_rescoring(clustered_sdf,
								n_cpus,
								column_name,
								software=software,
								rescoring_folder=rescoring_folder,
								protein_file=protein_file,
								pocket_definition=pocket_definition)

	# Assert the result
	assert isinstance(result, DataFrame)
	assert "Pose ID" in result.columns
	assert "PANTHER-Shape" in result.columns
	assert len(result) > 0
	assert all(result["PANTHER-Shape"].notna())
