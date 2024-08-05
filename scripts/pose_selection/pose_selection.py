import sys
import warnings
from pathlib import Path

import pandas as pd
from rdkit.Chem import PandasTools
from typing import Union, Optional

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.pose_selection.clustering.clustering_metrics.clustering_metrics import CLUSTERING_METRICS
from scripts.pose_selection.clustering.clustering import run_clustering
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS, rescore_docking
from scripts.utilities.logging import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def select_poses(poses: Union[Path, pd.DataFrame],
					selection_method: str,
					clustering_method: str,
					pocket_definition: dict,
					protein_file: Path,
					software: Path,
					n_cpus: int,
					output_file: Optional[Path] = None) -> pd.DataFrame:
	"""This function clusters all poses according to the metric selected using multiple CPU cores.

	Args:
		metric (str): A string representing the clustering metric to be used.
		method (str): A string representing the clustering method to be used.
		w_dir (str): A string representing the working directory.
		protein_file (str): A string representing the file path of the reference protein structure.
		poses (pandas.DataFrame): A pandas DataFrame containing the input data for clustering.
		n_cpus (int): An integer representing the number of CPU cores to be used for clustering.

	Returns:
		None. The function writes the clustered poses to a SDF file.
	"""

	# Process input
	if isinstance(poses, Path):
		poses_df = PandasTools.LoadSDF(poses, molColName='Molecule', idName='Pose ID')
	elif isinstance(poses, pd.DataFrame):
		poses_df = poses
	else:
		raise ValueError("poses must be a Path or DataFrame")

	# Get unique IDs from the input DataFrame
	printlog(f"*Calculating {selection_method} metrics and clustering*")
	# Add additional columns to the DataFrame for clustering
	poses_df["Pose_Number"] = poses_df["Pose ID"].str.split("_").str[2].astype(int)
	poses_df["Docking_program"] = poses_df["Pose ID"].str.split("_").str[1].astype(str)

	if selection_method == "bestpose":
		# Select the best pose for each ID and docking program
		min_pose_indices = poses_df.groupby(["ID", "Docking_program"])["Pose_Number"].idxmin()
		selected_poses = poses_df.loc[min_pose_indices]
	elif selection_method in [
		"bestpose_GNINA", "bestpose_SMINA", "bestpose_PLANTS", "bestpose_QVINAW", "bestpose_QVINA2"]:
		# Select the best pose for each ID based on the specified docking program
		min_pose_indices = poses_df.groupby(["ID", "Docking_program"])["Pose_Number"].idxmin()
		selected_poses = poses_df.loc[min_pose_indices]
		selected_poses = selected_poses[selected_poses["Docking_program"] == selection_method.split("_")[1]]
	elif selection_method in list(CLUSTERING_METRICS.keys()):
		# Perform clustering using multiple CPU cores
		selected_poses = run_clustering(poses_df, selection_method, clustering_method, protein_file, n_cpus)
	elif selection_method in list(RESCORING_FUNCTIONS.keys()):
		# Perform rescoring using the specified metric scoring function
		selected_poses = rescore_docking(poses_df, protein_file, pocket_definition, software, selection_method, n_cpus)
	else:
		raise ValueError(f"Invalid clustering metric: {selection_method}")
	# Clean up the Pose ID column
	selected_poses["Pose ID"] = selected_poses["Pose ID"].astype(str).replace("[()',]", "", regex=True)
	# Filter the original DataFrame based on the clustered poses
	filtered_poses = poses_df[poses_df["Pose ID"].isin(selected_poses["Pose ID"])]
	filtered_poses = filtered_poses[["Pose ID", "Molecule", "ID"]]
	# Write the filtered poses to a SDF file
	if output_file:
		PandasTools.WriteSDF(filtered_poses,
								str(output_file),
								molColName="Molecule",
								idName="Pose ID",
								properties=list(filtered_poses.columns))
		return output_file
	else:
		return filtered_poses
