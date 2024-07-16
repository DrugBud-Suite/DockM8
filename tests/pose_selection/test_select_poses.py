from pathlib import Path
import sys
import os

from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.pose_selection.pose_selection import select_poses


def test_select_poses():
	input_file = dockm8_path / "tests" / "test_files" / "pose_selection" / "allposes.sdf"
	protein_file = dockm8_path / "tests" / "test_files" / "pose_selection" / "4kd1_p.pdb"
	software_path = dockm8_path / "software"
	output_dir = dockm8_path / "tests" / "test_files" / "pose_selection"

	# Define a dummy pocket definition (adjust as needed)
	pocket_definition = {"center_x": 0, "center_y": 0, "center_z": 0, "size_x": 20, "size_y": 20, "size_z": 20}

	# Test RMSD clustering
	rmsd_output = output_dir / "RMSD_clustered.sdf"
	selected_poses_rmsd = select_poses(poses=input_file,
										selection_method="RMSD",
										clustering_method="KMedoids",
										pocket_definition=pocket_definition,
										protein_file=protein_file,
										software=software_path,
										n_cpus=int(os.cpu_count() * 0.9),
										output_file=rmsd_output)

	expected_rmsd_output = PandasTools.LoadSDF(str(dockm8_path / "tests" / "test_files" / "pose_selection" /
													"RMSD_clustered.sdf"),
												molColName="Molecule",
												idName="Pose ID")
	assert len(selected_poses_rmsd) == len(expected_rmsd_output)
	assert set(selected_poses_rmsd["Pose ID"]) == set(expected_rmsd_output["Pose ID"])

	# Test bestpose_GNINA selection
	gnina_output = output_dir / "bestpose_GNINA_clustered.sdf"
	selected_poses_gnina = select_poses(poses=input_file,
										selection_method="bestpose_GNINA",
										clustering_method="KMedoids",
										pocket_definition=pocket_definition,
										protein_file=protein_file,
										software=software_path,
										n_cpus=int(os.cpu_count() * 0.9),
										output_file=gnina_output)

	expected_gnina_output = PandasTools.LoadSDF(str(dockm8_path / "tests" / "test_files" / "pose_selection" /
													"bestpose_GNINA_clustered.sdf"),
												molColName="Molecule",
												idName="Pose ID")
	assert len(selected_poses_gnina) == len(expected_gnina_output)
	assert set(selected_poses_gnina["Pose ID"]) == set(expected_gnina_output["Pose ID"])

	# Test KORP-PL rescoring
	korpl_output = output_dir / "KORP-PL_clustered.sdf"
	selected_poses_korpl = select_poses(poses=input_file,
										selection_method="KORP-PL",
										clustering_method="KMedoids",
										pocket_definition=pocket_definition,
										protein_file=protein_file,
										software=software_path,
										n_cpus=int(os.cpu_count() * 0.9),
										output_file=korpl_output)

	expected_korpl_output = PandasTools.LoadSDF(str(dockm8_path / "tests" / "test_files" / "pose_selection" /
													"KORP-PL_clustered.sdf"),
												molColName="Molecule",
												idName="Pose ID")
	assert len(selected_poses_korpl) == len(expected_korpl_output)
	assert set(selected_poses_korpl["Pose ID"]) == set(expected_korpl_output["Pose ID"])
