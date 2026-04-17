import pytest
from pathlib import Path
from scripts.pose_selection.posebusters import bust_poses

def test_bust_poses_with_valid_files(tmp_path):
    # Use the provided test data files
    poses_path = Path("test_data/posebusters/gnina_docking_results.sdf")
    protein_file = Path("test_data/posebusters/prepared_protein.pdb")
    config_path = Path("scripts/pose_selection/posebusters_config.yml")
    output_path = tmp_path / "filtered_poses.sdf"

    # Call the function
    result = bust_poses(poses_path, protein_file, config_path, output_path)

    # Check if the output file was created
    assert result.exists()
    assert result == output_path

def test_bust_poses_with_invalid_poses_file(tmp_path):
    # Create a protein file only
    protein_file = tmp_path / "protein.pdb"
    config_path = Path("scripts/pose_selection/posebusters_config.yml")

    # Write dummy content to the protein file
    protein_file.write_text("Dummy protein content")

    # Call the function with a non-existent poses file
    with pytest.raises(FileNotFoundError):
        bust_poses(Path("non_existent_poses.sdf"), protein_file, config_path)

def test_bust_poses_with_invalid_protein_file(tmp_path):
    # Create a poses file only
    poses_path = tmp_path / "poses.sdf"
    config_path = Path("scripts/pose_selection/posebusters_config.yml")

    # Write dummy content to the poses file
    poses_path.write_text("Dummy poses content")

    # Call the function with a non-existent protein file
    with pytest.raises(FileNotFoundError):
        bust_poses(poses_path, Path("non_existent_protein.pdb"), config_path)

def test_bust_poses_with_invalid_config_file(tmp_path):
    # Use the provided test data files
    poses_path = Path("test_data/posebusters/gnina_docking_results.sdf")
    protein_file = Path("test_data/posebusters/prepared_protein.pdb")
    output_path = tmp_path / "filtered_poses.sdf"

    # Call the function with a non-existent config file
    with pytest.raises(FileNotFoundError):
        bust_poses(poses_path, protein_file, Path("non_existent_config.yml"), output_path)
