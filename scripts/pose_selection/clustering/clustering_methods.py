from pathlib import Path
import sys
import traceback
import warnings

import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.utilities.logging import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def kmedoids_S_clustering(input_dataframe: pd.DataFrame) -> pd.DataFrame:
	"""
    Applies k-medoids clustering to the input dataframe, which contains clustering metrics.
    Calculates the silhouette scores for different numbers of clusters and selects the optimal number of clusters
    based on the highest silhouette score. Then, it performs k-medoids clustering with the optimal number of clusters
    and returns the list of cluster centers and their corresponding Pose IDs.

    Args:
        input_dataframe: A dataframe containing clustering metrics. The columns represent different molecules
        and the rows represent different poses.

    Returns:
        A dataframe containing the Pose IDs of the cluster centers.
    """
	try:
		df = input_dataframe.copy()
		molecule_list = input_dataframe.columns.tolist()
		# Scale the values of the molecules in the dataframe using StandardScaler
		scaler = StandardScaler()
		df[molecule_list] = scaler.fit_transform(df)
		# Calculate silhouette average score for every cluster and select the optimal number of clusters
		silhouette_scores = {}
		for num_clusters in range(2, 5):
			kmedoids = KMedoids(n_clusters=num_clusters, method="pam", init="build", max_iter=150)
			kmedoids.fit_predict(df)
			silhouette_average_score = silhouette_score(df, kmedoids.labels_)
			silhouette_scores[num_clusters] = silhouette_average_score
		# Determine optimal number of clusters
		optimum_no_clusters = max(silhouette_scores, key=silhouette_scores.get)
		# Apply optimized k-medoids clustering
		kmedoids = KMedoids(n_clusters=optimum_no_clusters, method="pam", init="build", max_iter=150)
		clusters = kmedoids.fit_predict(df)
		df["KMedoids Cluster"] = clusters
		df["Pose ID"] = molecule_list
		# Determine cluster centers
		centroids = kmedoids.cluster_centers_
		cluster_centers = pd.DataFrame(centroids, columns=molecule_list)
		# Merge the dataframe with the cluster labels and the dataframe of cluster centers on the molecule list
		merged_df = pd.merge(df, cluster_centers, on=molecule_list, how="inner")
		merged_df = merged_df[["Pose ID"]]
		return merged_df
	except Exception as e:
		printlog(f"Error in kmedoids_S_clustering: {e}")
		traceback.print_exc()
		return None


def affinity_propagation_clustering(input_dataframe: pd.DataFrame) -> pd.DataFrame:
	"""
    Applies affinity propagation clustering to the input dataframe, which is a matrix of clustering metrics.
    Returns a dataframe containing the Pose IDs of the cluster centers.
    """
	df = input_dataframe.copy()
	molecule_list = df.columns.tolist()
	# Scale the clustering metrics
	scaler = StandardScaler()
	df[molecule_list] = scaler.fit_transform(df[molecule_list])
	# Apply affinity propagation clustering
	affinity_propagation = AffinityPropagation(max_iter=150)
	clusters = affinity_propagation.fit_predict(df)
	# Assign cluster labels and Pose IDs to the dataframe
	df["Affinity Cluster"] = clusters
	df["Pose ID"] = molecule_list
	# Determine cluster centers
	cluster_centers = pd.DataFrame(affinity_propagation.cluster_centers_, columns=molecule_list)
	# Merge dataframe with cluster centers based on the molecule list
	merged_df = pd.merge(df, cluster_centers, on=molecule_list, how="inner")
	# Select only the Pose ID column from the merged dataframe
	merged_df = merged_df[["Pose ID"]]
	return merged_df
