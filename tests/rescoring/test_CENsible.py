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

from scripts.rescoring.rescoring_functions.CENsible import CENsible_rescoring, check_and_download_censible, find_executable


@pytest.fixture
def test_data():
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	w_dir = dockm8_path / "tests/test_files/rescoring"
	protein_file = dockm8_path / "tests/test_files/rescoring/example_prepared_receptor_1fvv.pdb"
	clustered_sdf = dockm8_path / "tests/test_files/rescoring/example_poses_1fvv.sdf"
	n_cpus = int(os.cpu_count() * 0.9)
	return w_dir, protein_file, clustered_sdf, n_cpus


def test_CENsible_rescoring(test_data):
	# Define the input arguments for the function
	w_dir, protein_file, clustered_sdf, n_cpus = test_data
	column_name = "CENsible"
	rescoring_folder = w_dir / f"rescoring_{clustered_sdf.stem}"

	# Ensure CENsible is downloaded and set up
	censible_folder = check_and_download_censible()
	assert censible_folder.exists(), "CENsible folder does not exist after download attempt"

	# Call the function
	result_file = CENsible_rescoring(clustered_sdf,
										n_cpus,
										column_name,
										rescoring_folder=rescoring_folder,
										protein_file=protein_file)

	# Assert the result
	assert result_file.exists(), f"Result file does not exist: {result_file}"
	result = pd.read_csv(result_file)
	assert isinstance(result, DataFrame), "Result is not a DataFrame"
	assert "Pose ID" in result.columns, "Pose ID column is missing from the result"
	assert column_name in result.columns, f"{column_name} column is missing from the result"
	assert len(result) > 0, "Result DataFrame is empty"

	# Additional assertions specific to CENsible
	assert all(result[column_name].notna()), "Some CENsible scores are NaN"
