import sys
import warnings
from pathlib import Path

import pandas as pd
from rdkit.Chem import PandasTools

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


def select_poses(selection_method: str,
		clustering_method: str,
		w_dir: Path,
		protein_file: Path,
		software: Path,
		all_poses: pd.DataFrame,
		n_cpus: int):
	"""This function clusters all poses according to the metric selected using multiple CPU cores.

    Args:
        metric (str): A string representing the clustering metric to be used.
        method (str): A string representing the clustering method to be used.
        w_dir (str): A string representing the working directory.
        protein_file (str): A string representing the file path of the reference protein structure.
        all_poses (pandas.DataFrame): A pandas DataFrame containing the input data for clustering.
        n_cpus (int): An integer representing the number of CPU cores to be used for clustering.

    Returns:
        None. The function writes the clustered poses to a SDF file.
    """
	# Create a directory for clustering results
	cluster_dir = Path(w_dir) / "clustering"
	cluster_dir.mkdir(exist_ok=True)
	cluster_file = cluster_dir / f"{selection_method}_clustered.sdf"

	# Check if clustering has already been done for the given metric
	if not cluster_file.exists():
		# Get unique IDs from the input DataFrame
		printlog(f"*Calculating {selection_method} metrics and clustering*")
		# Add additional columns to the DataFrame for clustering
		all_poses["Pose_Number"] = all_poses["Pose ID"].str.split("_").str[2].astype(int)
		all_poses["Docking_program"] = all_poses["Pose ID"].str.split("_").str[1].astype(str)

		if selection_method == "bestpose":
			# Select the best pose for each ID and docking program
			min_pose_indices = all_poses.groupby(["ID", "Docking_program"])["Pose_Number"].idxmin()
			clustered_poses = all_poses.loc[min_pose_indices]
		elif selection_method in [
			"bestpose_GNINA", "bestpose_SMINA", "bestpose_PLANTS", "bestpose_QVINAW", "bestpose_QVINA2"]:
			# Select the best pose for each ID based on the specified docking program
			min_pose_indices = all_poses.groupby(["ID", "Docking_program"])["Pose_Number"].idxmin()
			clustered_poses = all_poses.loc[min_pose_indices]
			clustered_poses = clustered_poses[clustered_poses["Docking_program"] == selection_method.split("_")[1]]
		elif selection_method in CLUSTERING_METRICS.keys():
			# Perform clustering using multiple CPU cores
			clustered_poses = run_clustering(all_poses, selection_method, clustering_method, protein_file, n_cpus)
		elif selection_method in RESCORING_FUNCTIONS.keys():
			# Perform rescoring using the specified metric scoring function
			clustered_poses = rescore_docking(w_dir, protein_file, software, selection_method, n_cpus)
		else:
			raise ValueError(f"Invalid clustering metric: {selection_method}")
		# Clean up the Pose ID column
		clustered_poses["Pose ID"] = clustered_poses["Pose ID"].astype(str).replace("[()',]", "", regex=True)
		# Filter the original DataFrame based on the clustered poses
		filtered_poses = all_poses[all_poses["Pose ID"].isin(clustered_poses["Pose ID"])]
		filtered_poses = filtered_poses[["Pose ID", "Molecule", "ID"]]
		# Write the filtered poses to a SDF file
		PandasTools.WriteSDF(filtered_poses, str(cluster_file), molColName="Molecule", idName="Pose ID")
	else:
		printlog(f"Clustering using {selection_method} already done, moving to next metric...")
	return
