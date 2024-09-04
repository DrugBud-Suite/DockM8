import pytest
import os
import sys
from pathlib import Path
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.docking_postprocessing.minimization.ff_pose_minimization import minimize_poses, extract_pocket_from_coords
from scripts.docking_postprocessing.deeprmsd.deeprmsd import optimize_poses

@pytest.fixture
def common_test_data():
	"""Set up common test data."""
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	input_folder = dockm8_path / "tests/test_files/docking_postprocessing/ff_pose_minimization/"

	regular_input_sdf = input_folder / "plantain_docking_results.sdf"
	blind_input_sdf = input_folder / "fabind_docking_results.sdf"
	output_path = input_folder / "minimized_poses_output.sdf"
	protein_file = input_folder / "prepared_protein.pdb"
	n_cpus = max(1, int(os.cpu_count() * 0.9))

	return regular_input_sdf, blind_input_sdf, output_path, protein_file, n_cpus


@pytest.fixture
def cleanup(request):
	"""Cleanup fixture to remove generated files after each test."""
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	output_dir = dockm8_path / "tests/test_files/docking_postprocessing/ff_pose_minimization/"

	def remove_created_files():
		for file in output_dir.iterdir():
			if file.name == "minimized_poses_output.sdf":
				file.unlink()

	request.addfinalizer(remove_created_files)


def test_minimize_poses_mmff94_regular(common_test_data, cleanup):
	"""Test case for minimize_poses function using MMFF94 forcefield with regular docking."""
	regular_input_sdf, _, output_path, protein_file, n_cpus = common_test_data
	config = {
		'forcefield': 'MMFF94',
		'n_steps': 500,
		'distance_constraint': 0.5,
		'docking_type': 'regular',
		'pocket': {
			"center": [-9.67, 207.73, 113.41], "size": [20.0, 20.0, 20.0]}}

	result_df = minimize_poses(str(regular_input_sdf), str(protein_file), str(output_path), config, n_cpus=n_cpus)

	assert output_path.exists()
	assert result_df is not None
	assert len(result_df) > 0
	assert 'pose_energy' in result_df.columns
	assert 'minimized_pose_energy' in result_df.columns


def test_minimize_poses_mmff94s_regular(common_test_data, cleanup):
	"""Test case for minimize_poses function using MMFF94s forcefield with regular docking."""
	regular_input_sdf, _, output_path, protein_file, n_cpus = common_test_data
	config = {
		'forcefield': 'MMFF94s',
		'n_steps': 500,
		'distance_constraint': 0.5,
		'docking_type': 'regular',
		'pocket': {
			'center': [0, 0, 0], 'size': [10, 10, 10]}}

	result_df = minimize_poses(str(regular_input_sdf), str(protein_file), str(output_path), config, n_cpus=n_cpus)

	assert output_path.exists()
	assert result_df is not None
	assert len(result_df) > 0
	assert 'pose_energy' in result_df.columns
	assert 'minimized_pose_energy' in result_df.columns


def test_minimize_poses_uff_regular(common_test_data, cleanup):
	"""Test case for minimize_poses function using UFF forcefield with regular docking."""
	regular_input_sdf, _, output_path, protein_file, n_cpus = common_test_data
	config = {
		'forcefield': 'UFF',
		'n_steps': 500,
		'distance_constraint': 0.5,
		'docking_type': 'regular',
		'pocket': {
			'center': [0, 0, 0], 'size': [10, 10, 10]}}

	result_df = minimize_poses(str(regular_input_sdf), str(protein_file), str(output_path), config, n_cpus=n_cpus)

	assert output_path.exists()
	assert result_df is not None
	assert len(result_df) > 0
	assert 'pose_energy' in result_df.columns
	assert 'minimized_pose_energy' in result_df.columns


def test_minimize_poses_blind_docking(common_test_data):
	"""Test case for minimize_poses function with blind docking."""
	_, blind_input_sdf, output_path, protein_file, n_cpus = common_test_data
	config = {
		'forcefield': 'MMFF94s',
		'n_steps': 2000,
		'distance_constraint': 1,
		'docking_type': 'blind',
		'distance_from_ligand': 5.0}

	result_df = minimize_poses(str(blind_input_sdf), str(protein_file), str(output_path), config, n_cpus=n_cpus)

	assert output_path.exists()
	assert result_df is not None
	assert len(result_df) > 0
	assert 'pose_energy' in result_df.columns
	assert 'minimized_pose_energy' in result_df.columns


def test_minimize_poses_invalid_forcefield(common_test_data, cleanup):
	"""Test case for minimize_poses function with an invalid forcefield."""
	regular_input_sdf, _, output_path, protein_file, n_cpus = common_test_data
	config = {
		'forcefield': 'INVALID',
		'n_steps': 500,
		'distance_constraint': 0.5,
		'docking_type': 'regular',
		'pocket': {
			'center': [0, 0, 0], 'size': [10, 10, 10]}}

	result_df = minimize_poses(str(regular_input_sdf), str(protein_file), str(output_path), config, n_cpus=n_cpus)

	assert result_df is None


def test_minimize_poses_dataframe_input(common_test_data, cleanup):
	"""Test case for minimize_poses function with DataFrame input."""
	regular_input_sdf, _, output_path, protein_file, n_cpus = common_test_data
	config = {
		'forcefield': 'MMFF94',
		'n_steps': 500,
		'distance_constraint': 0.5,
		'docking_type': 'regular',
		'pocket': {
			'center': [0, 0, 0], 'size': [10, 10, 10]}}

	input_df = PandasTools.LoadSDF(str(regular_input_sdf))
	result_df = minimize_poses(input_df, str(protein_file), str(output_path), config, n_cpus=n_cpus)

	assert output_path.exists()
	assert result_df is not None
	assert len(result_df) > 0
	assert 'pose_energy' in result_df.columns
	assert 'minimized_pose_energy' in result_df.columns


def test_minimize_poses_invalid_input(common_test_data, cleanup):
	"""Test case for minimize_poses function with invalid input type."""
	_, _, output_path, protein_file, n_cpus = common_test_data
	config = {
		'forcefield': 'MMFF94',
		'n_steps': 500,
		'distance_constraint': 0.5,
		'docking_type': 'regular',
		'pocket': {
			'center': [0, 0, 0], 'size': [10, 10, 10]}}

	invalid_input = [1, 2, 3]   # Neither string nor DataFrame
	result_df = minimize_poses(invalid_input, str(protein_file), str(output_path), config, n_cpus=n_cpus)

	assert result_df is None


def test_minimize_poses_missing_molecule_column(common_test_data, cleanup):
	"""Test case for minimize_poses function with DataFrame missing 'Molecule' column."""
	_, _, output_path, protein_file, n_cpus = common_test_data
	config = {
		'forcefield': 'MMFF94',
		'n_steps': 500,
		'distance_constraint': 0.5,
		'docking_type': 'regular',
		'pocket': {
			'center': [0, 0, 0], 'size': [10, 10, 10]}}

	invalid_df = pd.DataFrame({'A': [1, 2, 3]})
	result_df = minimize_poses(invalid_df, str(protein_file), str(output_path), config, n_cpus=n_cpus)

	assert result_df is None

def test_deeprmsd(common_test_data):
	regular_input_sdf, _, output_path, protein_file, n_cpus = common_test_data
	dockm8_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None).parent
	output_path = dockm8_path / "tests" / "test_files" / "docking_postprocessing" / "deeprmsd"
	result_df = optimize_poses(regular_input_sdf, protein_file, output_path, dockm8_path / "software")
