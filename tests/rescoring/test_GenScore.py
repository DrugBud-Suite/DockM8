import sys
from pathlib import Path

import pytest
from pandas import DataFrame
from rdkit import Chem

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring_functions.GenScore import GenScore


@pytest.fixture
def test_data():
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	protein_file = dockm8_path / "tests/test_files/rescoring/example_prepared_receptor_1fvv.pdb"
	software = dockm8_path / "software"
	clustered_sdf = dockm8_path / "tests/test_files/rescoring/example_poses_1fvv.sdf"
	n_cpus = 1
	output_dir = dockm8_path / "tests/test_files/rescoring/output"
	output_dir.mkdir(exist_ok=True)
	return protein_file, software, clustered_sdf, n_cpus, output_dir


@pytest.mark.parametrize("score_type", ["scoring", "docking", "balanced"])
def test_GenScore_rescoring(test_data, score_type):
	protein_file, software, clustered_sdf, n_cpus, output_dir = test_data

	genscore = GenScore(score_type, software)

	result = genscore.rescore(clustered_sdf, n_cpus, software=software, protein_file=protein_file)

	assert isinstance(result, DataFrame)
	assert "Pose ID" in result.columns
	assert genscore.column_name in result.columns
	assert len(result) > 0

	suppl = Chem.SDMolSupplier(str(clustered_sdf))
	num_molecules = len([mol for mol in suppl if mol is not None])
	assert len(result) == num_molecules

	output_file = output_dir / f"{genscore.column_name}_scores.csv"
	result.to_csv(output_file, index=False)
	assert output_file.is_file()
