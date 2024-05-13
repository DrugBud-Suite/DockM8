import pytest
import os
import sys
from pathlib import Path

from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
dockm8_path = next(
    (p / "DockM8" for p in Path(__file__).resolve().parents if (p / "DockM8").is_dir()),
    None,
)
sys.path.append(str(dockm8_path))

from scripts.docking_postprocessing.docking_postprocessing import docking_postprocessing

@pytest.fixture
def common_test_data():
    """Set up common test data."""
    input_sdf = Path(dockm8_path / "tests/test_files/docking_postprocessing/example_poses_1fvv.sdf")
    output_path = Path(dockm8_path / "tests/test_files/docking_postprocessing/example_poses_1fvv_postprocessed.sdf")
    protein_file = Path(dockm8_path / "tests/test_files/docking_postprocessing/example_prepared_receptor_1fvv.pdb")
    bust_poses = True
    strain_cutoff = 5
    clash_cutoff = 3
    n_cpus = int(os.cpu_count()*0.9)
    return input_sdf, output_path, protein_file, bust_poses, strain_cutoff, clash_cutoff, n_cpus

def test_docking_postprocessing(common_test_data):
    """
    Test case for docking_postprocessing function.

    Args:
        common_test_data: A tuple containing the input_sdf, output_path, protein_file, bust_poses, strain_cutoff, clash_cutoff, and n_cpus.

    Returns:
        None
    """
    input_sdf, output_path, protein_file, bust_poses, strain_cutoff, clash_cutoff, n_cpus = common_test_data
    result = docking_postprocessing(input_sdf, output_path, protein_file, bust_poses, strain_cutoff, clash_cutoff, n_cpus)
    assert result == output_path
    assert output_path.exists()
    output_data = PandasTools.LoadSDF(str(output_path))
    assert len(output_data) == 14
    

def test_docking_postprocessing_without_pose_busting(common_test_data):
    """
    Test case for docking_postprocessing function without pose busting.

    Args:
        common_test_data: A tuple containing the input_sdf, output_path, protein_file, bust_poses, strain_cutoff, clash_cutoff, and n_cpus.

    Returns:
        None
    """
    input_sdf, output_path, protein_file, bust_poses, strain_cutoff, clash_cutoff, n_cpus = common_test_data
    bust_poses = False
    result = docking_postprocessing(input_sdf, output_path, protein_file, bust_poses, strain_cutoff, clash_cutoff, n_cpus)
    assert result == output_path
    assert output_path.exists()

def test_docking_postprocessing_with_no_cutoffs(common_test_data):
    """
    Test case for docking_postprocessing function with no strain or clash cutoffs.

    Args:
        common_test_data: A tuple containing the input_sdf, output_path, protein_file, bust_poses, strain_cutoff, clash_cutoff, and n_cpus.

    Returns:
        None
    """
    input_sdf, output_path, protein_file, bust_poses, strain_cutoff, clash_cutoff, n_cpus = common_test_data
    strain_cutoff = None
    clash_cutoff = None
    result = docking_postprocessing(input_sdf, output_path, protein_file, bust_poses, strain_cutoff, clash_cutoff, n_cpus)
    assert result == output_path
    assert output_path.exists()
    
def remove_test_data(output_file):
    if output_file.is_file():
        output_file.unlink()
    else:
        pass

remove_test_data(Path(dockm8_path / "tests/test_files/example_poses_1fvv_postprocessed.sdf"))