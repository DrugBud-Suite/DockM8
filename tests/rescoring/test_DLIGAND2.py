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

from scripts.rescoring.rescoring_functions.DLIGAND2 import DLIGAND2_rescoring


@pytest.fixture
def test_data():
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	w_dir = dockm8_path / "tests/test_files/rescoring"
	protein_file = dockm8_path / "tests/test_files/rescoring/example_prepared_receptor_1fvv.pdb"
	clustered_sdf = dockm8_path / "tests/test_files/rescoring/example_poses_1fvv.sdf"
	n_cpus = int(os.cpu_count() * 0.9)
	return w_dir, protein_file, clustered_sdf, n_cpus


def test_DLIGAND2_rescoring(test_data):
	# Define the input arguments for the function
	w_dir, protein_file, clustered_sdf, n_cpus = test_data
	column_name = "DLIGAND2"
	rescoring_folder = w_dir / f"rescoring_{clustered_sdf.stem}"
	etype = 1                   # You can change this to 2 if you want to test the other potential type

	# Call the function
	result_file = DLIGAND2_rescoring(clustered_sdf,
										n_cpus,
										column_name,
										rescoring_folder=rescoring_folder,
										protein_file=protein_file,
										etype=etype)

	# Assert the result
	assert result_file.exists()
	result = pd.read_csv(result_file)
	assert isinstance(result, DataFrame)
	assert "Pose ID" in result.columns
	assert "DLIGAND2" in result.columns
	assert len(result) > 0
	assert all(result["DLIGAND2"].notna())
