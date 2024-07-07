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

from scripts.rescoring.rescoring_functions.GenScore import GenScore_rescoring


@pytest.fixture
def test_data():
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	w_dir = dockm8_path / "tests/test_files/rescoring"
	software = dockm8_path / "software"
	protein_file = dockm8_path / "tests/test_files/rescoring/example_prepared_receptor_1fvv.pdb"
	clustered_sdf = dockm8_path / "tests/test_files/rescoring/example_poses_1fvv.sdf"
	n_cpus = 2
	return w_dir, software, protein_file, clustered_sdf, n_cpus


@pytest.mark.parametrize("column_name", ["GenScore-scoring", "GenScore-docking", "GenScore-balanced"])
def test_GenScore_rescoring(test_data, column_name):
	# Define the input arguments for the function
	w_dir, software, protein_file, clustered_sdf, n_cpus = test_data
	rescoring_folder = w_dir / f"rescoring_{clustered_sdf.stem}"

	# Call the function
	result = GenScore_rescoring(clustered_sdf,
								1,
								column_name,
								software=software,
								rescoring_folder=rescoring_folder,
								protein_file=protein_file)

	# Assert the result
	assert isinstance(result, DataFrame)
	assert "Pose ID" in result.columns
	assert column_name in result.columns
	assert len(result) > 0
	assert all(result[column_name].notna())

	# Check if the output file was created and then deleted
	assert not (rescoring_folder / f"{column_name}_rescoring" / f"{column_name}.csv").exists()
	assert (rescoring_folder / f"{column_name}_rescoring" / f"{column_name}_scores.csv").exists()
