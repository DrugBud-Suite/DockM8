import itertools
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
import pebble
from tqdm import tqdm

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.pose_selection.clustering.clustering_metrics.clustering_metrics import CLUSTERING_METRICS
from scripts.pose_selection.clustering.clustering_methods import affinity_propagation_clustering
from scripts.pose_selection.clustering.clustering_methods import kmedoids_S_clustering
from scripts.utilities.logging import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def calculate_and_cluster(df: pd.DataFrame, selection_method: str, clustering_method: str,
	protein_file: str) -> pd.DataFrame:
	"""
    Calculates a clustering metric and performs clustering on a given dataframe.

    Args:
        metric: A string representing the clustering metric to be used for calculation.
        method: A string representing the clustering method to be used for clustering.
        df: A pandas DataFrame containing the input data for clustering.
        protein_file: A string representing the file path of the reference protein structure.

    Returns:
        clustered_df: A pandas DataFrame containing the Pose IDs of the cluster centers.
    """
	# Dictionary mapping clustering methods to their corresponding clustering functions
	clustering_methods: Dict[str, Callable] = {
		"KMedoids": kmedoids_S_clustering, "AffProp": affinity_propagation_clustering, }
	# Generate all possible combinations of molecules in the dataframe
	subsets = np.array(list(itertools.combinations(df["Molecule"], 2)))
	# Create a dictionary mapping molecule names to their indices in the dataframe
	indices = {mol: idx for idx, mol in enumerate(df["Molecule"].values)}
	# Select the appropriate clustering metric function based on the input metric
	if selection_method == "3DScore":
		metric_func = CLUSTERING_METRICS["spyRMSD"]["function"]
	elif selection_method in CLUSTERING_METRICS.keys():
		metric_func = CLUSTERING_METRICS[selection_method]["function"]
	else:
		raise ValueError(f"Invalid metric '{selection_method}'")

	# Vectorize the metric calculation function for efficient computation
	vectorized_calc_vec = np.vectorize(lambda x, y: metric_func(x, y, protein_file))

	# Calculate the clustering metric values for all molecule pairs
	results = vectorized_calc_vec(subsets[:, 0], subsets[:, 1])

	# Map the molecule names to their corresponding indices in the dataframe
	i = np.array([indices[x] for x in subsets[:, 0]])
	j = np.array([indices[y] for y in subsets[:, 1]])

	# Create a matrix to store the pairwise clustering metric values
	matrix = np.zeros((len(df), len(df)))
	matrix[i, j] = results
	matrix[j, i] = results

	# Perform clustering based on the selected metric and method
	if selection_method == "3DScore":
		# If 3DScore is selected, calculate the sum of spyRMSD values for each molecule and select the molecule with the lowest sum
		clustered_df = pd.DataFrame(matrix, index=df["Pose ID"].values.tolist(), columns=df["Pose ID"].values.tolist())
		clustered_df["3DScore"] = clustered_df.sum(axis=1)
		clustered_df.sort_values(by="3DScore", ascending=True, inplace=True)
		clustered_df = clustered_df.head(1)
		clustered_df = pd.DataFrame(clustered_df.index, columns=["Pose ID"])
		clustered_df["Pose ID"] = clustered_df["Pose ID"].astype(str).str.replace("[()',]", "", regex=False)
		return clustered_df
	else:
		# For other clustering metrics, pass the matrix to the corresponding clustering method function
		matrix_df = pd.DataFrame(matrix, index=df["Pose ID"].values.tolist(), columns=df["Pose ID"].values.tolist())
		matrix_df.fillna(0)
		clustered_df = clustering_methods[clustering_method](matrix_df)
		return clustered_df


def run_clustering(all_poses: pd.DataFrame,
	selection_method: str,
	clustering_method: str,
	protein_file: Path,
	n_cpus: int) -> pd.DataFrame:
	"""
    Runs the clustering process on the input DataFrame using multiple CPU cores.

    Args:
        all_poses: A pandas DataFrame containing the input data for clustering.
        metric: A string representing the clustering metric to be used.
        method: A string representing the clustering method to be used.
        protein_file: A string representing the file path of the reference protein structure.
        n_cpus: An integer representing the number of CPU cores to be used for clustering.

    Returns:
        clustered_poses: A pandas DataFrame containing the Pose IDs of the cluster centers.
    """
	id_list = np.unique(np.array(all_poses["ID"]))
	clustered_dataframes = []
	with pebble.ProcessPool(max_workers=n_cpus) as executor:
		jobs = []
		for current_id in tqdm(id_list, desc=f"Submitting {selection_method} jobs...", unit="IDs"):
			try:
				# Schedule the clustering job for each ID
				job = executor.schedule(calculate_and_cluster,
					args=(all_poses[all_poses["ID"] == current_id],
					selection_method,
					clustering_method,
					protein_file,
					),
					timeout=120,
					)
				jobs.append(job)
			except pebble.TimeoutError as e:
				printlog("Timeout error in pebble job creation: " + str(e))
			except pebble.JobCancellationError as e:
				printlog("Job cancellation error in pebble job creation: " + str(e))
			except pebble.JobSubmissionError as e:
				printlog("Job submission error in pebble job creation: " + str(e))
			except Exception as e:
				printlog("Other error in pebble job creation: " + str(e))
		for job in tqdm(jobs, total=len(id_list), desc=f"Running {selection_method} clustering...", unit="jobs"):
			try:
				# Get the clustering results for each job
				res = job.result()
				clustered_dataframes.append(res)
			except Exception as e:
				printlog("Error in pebble job execution: " + str(e))
				pass
	clustered_poses = pd.concat(clustered_dataframes)
	return clustered_poses
