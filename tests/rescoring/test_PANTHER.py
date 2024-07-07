import os
from pathlib import Path
import sys

from pandas import DataFrame

import pytest

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring_functions.PANTHER import PANTHER


@pytest.fixture
def test_data():
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	w_dir = dockm8_path / "tests/test_files/rescoring"
	protein_file = dockm8_path / "tests/test_files/rescoring/example_prepared_receptor_1fvv.pdb"
	pocket_definition = {"center": [-9.67, 207.73, 113.41], "size": [20.0, 20.0, 20.0]}
	software = dockm8_path / "software"
	clustered_sdf = dockm8_path / "tests/test_files/rescoring/example_poses_1fvv.sdf"
	n_cpus = int(os.cpu_count() * 0.9)
	return w_dir, protein_file, pocket_definition, software, clustered_sdf, n_cpus


@pytest.mark.parametrize("score_type", ["PANTHER", "PANTHER-ESP", "PANTHER-Shape"])
def test_PANTHER_rescoring(test_data, score_type):
	w_dir, protein_file, pocket_definition, software, clustered_sdf, n_cpus = test_data
	rescoring_folder = w_dir / f"rescoring_{clustered_sdf.stem}"

	panther = PANTHER(score_type)

	result = panther.rescore(clustered_sdf,
								n_cpus,
								rescoring_folder=rescoring_folder,
								software=software,
								protein_file=protein_file,
								pocket_definition=pocket_definition)

	assert isinstance(result, DataFrame)
	assert "Pose ID" in result.columns
	assert panther.column_name in result.columns
	assert len(result) > 0
