import pytest
from pathlib import Path
import yaml
import pandas as pd
import sys
import os

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from dockm8_classes import DockM8Base
from scripts.utilities.config_parser import DockM8Error

TEST_FILES_DIR = Path(__file__).parent.parent / "test_files" / "run"


class ConcreteDockM8(DockM8Base):

	def run(self):
		pass   # Implement a dummy run method


@pytest.fixture
def dockm8_base():
	config_path = TEST_FILES_DIR / "test_config_standard.yml"
	return ConcreteDockM8(config_path)


def test_init(dockm8_base):
	assert dockm8_base.config is not None
	assert dockm8_base.software == dockm8_path / "software"
	assert dockm8_base.n_cpus == int(os.cpu_count() * 0.9)


def test_setup_working_directory(dockm8_base):
	w_dir = dockm8_base._setup_working_directory()
	assert w_dir.is_dir()
	assert w_dir.name == Path(dockm8_base.config['receptor(s)'][0]).stem


def test_prepare_protein(dockm8_base):
	prepared_protein = dockm8_base.prepare_protein()
	assert prepared_protein.is_file()
	assert prepared_protein.suffix == '.pdb'


def test_determine_pocket(dockm8_base):
	prepared_receptor = TEST_FILES_DIR / "test_prepared.pdb"
	pocket_definition = dockm8_base.determine_pocket(prepared_receptor)
	assert isinstance(pocket_definition, dict)
	assert 'center' in pocket_definition
	assert 'size' in pocket_definition


def test_prepare_ligands(dockm8_base):
	output_dir = dockm8_base.w_dir
	prepared_library = dockm8_base.prepare_ligands(TEST_FILES_DIR / "test_library.sdf", output_dir)
	assert prepared_library.is_file()
	assert prepared_library.suffix == '.sdf'


def test_run_single_docking(dockm8_base):
	prepared_receptor = TEST_FILES_DIR / "test_prepared.pdb"
	pocket_definition = dockm8_base.determine_pocket(prepared_receptor)
	prepared_library = dockm8_base.prepare_ligands(TEST_FILES_DIR / "test_library.sdf", dockm8_base.w_dir)

	all_poses_path, docked_poses = dockm8_base._run_single_docking(
		prepared_library, prepared_receptor, pocket_definition, dockm8_base.w_dir
	)

	assert all_poses_path.is_file()
	assert all_poses_path.suffix == '.sdf'
	assert isinstance(docked_poses, dict)
	for program, path in docked_poses.items():
		assert path.is_file()
		assert path.suffix == '.sdf'


def test_post_process(dockm8_base):
	prepared_receptor = TEST_FILES_DIR / "test_prepared.pdb"
	all_poses_path = TEST_FILES_DIR / "all_poses.sdf"
	processed_poses = dockm8_base.post_process(all_poses_path, prepared_receptor, dockm8_base.w_dir)
	assert processed_poses.is_file()
	assert processed_poses.suffix == '.sdf'


def test_select_poses(dockm8_base):
	prepared_receptor = dockm8_base.prepare_protein()
	pocket_definition = dockm8_base.determine_pocket(prepared_receptor)
	all_poses_path, _ = dockm8_base._run_single_docking(
		TEST_FILES_DIR / "test_library.sdf",
		prepared_receptor,
		pocket_definition,
		dockm8_base.w_dir
	)
	processed_poses = dockm8_base.post_process(all_poses_path, prepared_receptor, dockm8_base.w_dir)

	selected_poses = dockm8_base.select_poses(processed_poses, prepared_receptor, pocket_definition, dockm8_base.w_dir)
	assert isinstance(selected_poses, dict)
	for method, path in selected_poses.items():
		assert path.is_file()
		assert path.suffix == '.sdf'


def test_rescore_poses(dockm8_base):
	prepared_receptor = dockm8_base.prepare_protein()
	pocket_definition = dockm8_base.determine_pocket(prepared_receptor)
	all_poses_path, _ = dockm8_base._run_single_docking(
		TEST_FILES_DIR / "test_library.sdf",
		prepared_receptor,
		pocket_definition,
		dockm8_base.w_dir
	)
	processed_poses = dockm8_base.post_process(all_poses_path, prepared_receptor, dockm8_base.w_dir)
	selected_poses = dockm8_base.select_poses(processed_poses, prepared_receptor, pocket_definition, dockm8_base.w_dir)

	rescored_poses = dockm8_base.rescore_poses(selected_poses, prepared_receptor, pocket_definition, dockm8_base.w_dir)
	assert isinstance(rescored_poses, dict)
	for method, path in rescored_poses.items():
		assert path.is_file()
		assert path.suffix == '.csv'


def test_apply_consensus(dockm8_base):
	# First, we need to generate some rescored poses
	prepared_receptor = dockm8_base.prepare_protein()
	pocket_definition = dockm8_base.determine_pocket(prepared_receptor)
	all_poses_path, _ = dockm8_base._run_single_docking(
		TEST_FILES_DIR / "test_library.sdf",
		prepared_receptor,
		pocket_definition,
		dockm8_base.w_dir
	)
	processed_poses = dockm8_base.post_process(all_poses_path, prepared_receptor, dockm8_base.w_dir)
	selected_poses = dockm8_base.select_poses(processed_poses, prepared_receptor, pocket_definition, dockm8_base.w_dir)
	rescored_poses = dockm8_base.rescore_poses(selected_poses, prepared_receptor, pocket_definition, dockm8_base.w_dir)

	# Now we can test the apply_consensus method
	for method, rescored_file in rescored_poses.items():
		consensus_result = dockm8_base.apply_consensus(rescored_file, method)
		assert consensus_result.is_file()
		assert consensus_result.suffix == '.csv'


def test_create_batches(dockm8_base):
	input_file = TEST_FILES_DIR / "test_library.sdf"
	batches = dockm8_base.create_batches(input_file)
	assert isinstance(batches, list)
	assert all(batch.is_file() for batch in batches)


def test_process_batch(dockm8_base):
	batch = TEST_FILES_DIR / "test_library.sdf"
	prepared_receptor = dockm8_base.prepare_protein()
	pocket_definition = dockm8_base.determine_pocket(prepared_receptor)

	result = dockm8_base.process_batch(batch, prepared_receptor, pocket_definition)

	assert len(result) == 6
	prepared_batch, all_poses_path, docked_poses, processed_poses, selected_poses, rescored_poses = result

	assert prepared_batch.is_file()
	assert all_poses_path.is_file()
	assert isinstance(docked_poses, dict)
	assert processed_poses.is_file()
	assert isinstance(selected_poses, dict)
	assert isinstance(rescored_poses, dict)


# Add more tests as needed for other methods in DockM8Base


def test_invalid_config():
	invalid_config = {}         # An empty config should be invalid
	with pytest.raises(DockM8Error):
		DockM8Base(invalid_config)


# You can add more negative test cases to check how the class handles errors
