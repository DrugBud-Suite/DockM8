import os
import sys
from pathlib import Path

import pytest
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.docking.panther_docking import PantherDocking


@pytest.fixture
def test_data():
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	protein_file = dockm8_path / "tests/test_files/docking/prepared_protein.pdb"
	library_file = dockm8_path / "tests/test_files/docking/prepared_library.sdf"
	ligand_file = dockm8_path / "tests/test_files/docking/prepared_ligand.sdf"
	software = dockm8_path / "software"
	n_cpus = 2
	output_dir = dockm8_path / "tests/test_files/docking/output"
	output_dir.mkdir(exist_ok=True)
	pocket_definition = {"center": [-9.67, 207.73, 113.41], "size": [20.0, 20.0, 20.0]}
	return protein_file, library_file, ligand_file, software, n_cpus, output_dir, pocket_definition


def test_PantherDocking(test_data):
	protein_file, library_file, ligand_file, software, n_cpus, output_dir, pocket_definition = test_data

	panther = PantherDocking(software)

	# Test the full docking process with the library
	output_sdf = output_dir / "panther_docking_results.sdf"
	docking_results = panther.dock(library=library_file,
									protein_file=protein_file,
									pocket_definition=pocket_definition,
									exhaustiveness=8,
									n_poses=10,
									n_cpus=n_cpus,
									output_sdf=output_sdf)

	assert isinstance(docking_results, pd.DataFrame)
	assert "Pose ID" in docking_results.columns
	assert "PANTHER_Score" in docking_results.columns
	assert "ID" in docking_results.columns
	assert "Molecule" in docking_results.columns
	assert len(docking_results) > 0

	# Check if the number of poses matches the input ligands in the library
	library_suppl = Chem.SDMolSupplier(str(library_file))
	num_library_ligands = len([mol for mol in library_suppl if mol is not None])
	assert len(docking_results) <= num_library_ligands * 10             # 10 poses per ligand at maximum

	# Check if the output SDF file was created
	assert output_sdf.is_file()

	# Verify the content of the output SDF file
	output_df = PandasTools.LoadSDF(str(output_sdf), molColName="Molecule", idName="Pose ID")
	assert "Pose ID" in output_df.columns
	assert "PANTHER_Score" in output_df.columns
	assert "ID" in output_df.columns
	assert "Molecule" in output_df.columns
	assert len(output_df) == len(docking_results)