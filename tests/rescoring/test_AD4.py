import os
from pathlib import Path
import sys

from pandas import DataFrame
import pytest
from rdkit import Chem

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.rescoring.rescoring_functions.AD4 import AD4


@pytest.fixture
def test_data():
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	protein_file = dockm8_path / "tests/test_files/rescoring/example_prepared_receptor_1fvv.pdb"
	software = dockm8_path / "software"
	clustered_sdf = dockm8_path / "tests/test_files/rescoring/example_poses_1fvv.sdf"
	n_cpus = int(os.cpu_count() * 0.9)
	output_dir = dockm8_path / "tests/test_files/rescoring/output"
	output_dir.mkdir(exist_ok=True)
	return protein_file, software, clustered_sdf, n_cpus, output_dir


def test_AD4_rescoring(test_data):
	# Define the input arguments for the function
	protein_file, software, clustered_sdf, n_cpus, output_dir = test_data

	# Initialize the AD4 class
	ad4 = AD4()

	# Call the function
	result = ad4.rescore(clustered_sdf, n_cpus, software=software, protein_file=protein_file)

	# Assert the result
	assert isinstance(result, DataFrame)
	assert "Pose ID" in result.columns
	assert ad4.column_name in result.columns
	assert len(result) > 0

	# Check if the number of scores matches the number of molecules in the input SDF
	suppl = Chem.SDMolSupplier(str(clustered_sdf))
	num_molecules = len([mol for mol in suppl if mol is not None])
	assert len(result) == num_molecules

	# Write the result DataFrame as a CSV file
	output_file = output_dir / f"{ad4.column_name}_scores.csv"
	result.to_csv(output_file, index=False)
	assert output_file.is_file()
