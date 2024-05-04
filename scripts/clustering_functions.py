import itertools
import sys
import traceback
import warnings
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd
import pebble
from rdkit.Chem import PandasTools
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from tqdm import tqdm

# Search for 'DockM8' in parent directories
dockm8_path = next((p / 'DockM8' for p in Path(__file__).resolve().parents if (p / 'DockM8').is_dir()), None)
sys.path.append(str(dockm8_path))

from scripts.clustering_metrics import CLUSTERING_METRICS
from scripts.rescoring_functions import RESCORING_FUNCTIONS, rescore_docking
from scripts.utilities import printlog

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
            kmedoids = KMedoids(n_clusters=num_clusters,
                                method='pam',
                                init='build',
                                max_iter=150)
            kmedoids.fit_predict(df)
            silhouette_average_score = silhouette_score(df, kmedoids.labels_)
            silhouette_scores[num_clusters] = silhouette_average_score
        #Determine optimal number of clusters
        optimum_no_clusters = max(silhouette_scores, key=silhouette_scores.get)
        # Apply optimized k-medoids clustering
        kmedoids = KMedoids(n_clusters=optimum_no_clusters,
                            method='pam',
                            init='build',
                            max_iter=150)
        clusters = kmedoids.fit_predict(df)
        df['KMedoids Cluster'] = clusters
        df['Pose ID'] = molecule_list
        # Determine cluster centers
        centroids = kmedoids.cluster_centers_
        cluster_centers = pd.DataFrame(centroids, columns=molecule_list)
        # Merge the dataframe with the cluster labels and the dataframe of cluster centers on the molecule list
        merged_df = pd.merge(df, cluster_centers, on=molecule_list, how='inner')
        merged_df = merged_df[['Pose ID']]
        return merged_df
    except Exception as e:
        print(f"Error in kmedoids_S_clustering: {e}")
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
    df['Affinity Cluster'] = clusters
    df['Pose ID'] = molecule_list
    # Determine cluster centers
    cluster_centers = pd.DataFrame(affinity_propagation.cluster_centers_, columns=molecule_list)
    # Merge dataframe with cluster centers based on the molecule list
    merged_df = pd.merge(df, cluster_centers, on=molecule_list, how='inner')
    # Select only the Pose ID column from the merged dataframe
    merged_df = merged_df[['Pose ID']]
    return merged_df

def calculate_and_cluster(clustering_metric: str, clustering_method: str, df: pd.DataFrame, protein_file: str) -> pd.DataFrame:
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
    clustering_methods: Dict[str, Callable] = {'KMedoids': kmedoids_S_clustering,
                                    'AffProp': affinity_propagation_clustering}
    # Generate all possible combinations of molecules in the dataframe
    subsets = np.array(list(itertools.combinations(df['Molecule'], 2)))
    # Create a dictionary mapping molecule names to their indices in the dataframe
    indices = {mol: idx for idx, mol in enumerate(df['Molecule'].values)}
    # Select the appropriate clustering metric function based on the input metric
    if clustering_metric == '3DScore':
        metric_func = CLUSTERING_METRICS['spyRMSD']['function']
    elif clustering_metric in CLUSTERING_METRICS.keys():
        metric_func = CLUSTERING_METRICS[clustering_metric]['function']
    else:
        raise ValueError(f"Invalid metric '{clustering_metric}'")

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
    if clustering_metric == '3DScore':
        # If 3DScore is selected, calculate the sum of spyRMSD values for each molecule and select the molecule with the lowest sum
        clustered_df = pd.DataFrame(matrix,
                                    index=df['Pose ID'].values.tolist(),
                                    columns=df['Pose ID'].values.tolist())
        clustered_df['3DScore'] = clustered_df.sum(axis=1)
        clustered_df.sort_values(by='3DScore', ascending=True, inplace=True)
        clustered_df = clustered_df.head(1)
        clustered_df = pd.DataFrame(clustered_df.index, columns=['Pose ID'])
        clustered_df['Pose ID'] = clustered_df['Pose ID'].astype(str).str.replace('[()\',]', '', regex=False)
        return clustered_df
    else:
        # For other clustering metrics, pass the matrix to the corresponding clustering method function
        matrix_df = pd.DataFrame(matrix,
                                 index=df['Pose ID'].values.tolist(),
                                 columns=df['Pose ID'].values.tolist())
        matrix_df.fillna(0)
        clustered_df = clustering_methods[clustering_method](matrix_df)
        return clustered_df


def select_poses(selection_method : str, clustering_method : str, w_dir : Path, protein_file: Path, pocket_definition: dict, software: Path, all_poses : pd.DataFrame, ncpus : int):
    '''This function clusters all poses according to the metric selected using multiple CPU cores.

    Args:
        metric (str): A string representing the clustering metric to be used.
        method (str): A string representing the clustering method to be used.
        w_dir (str): A string representing the working directory.
        protein_file (str): A string representing the file path of the reference protein structure.
        all_poses (pandas.DataFrame): A pandas DataFrame containing the input data for clustering.
        ncpus (int): An integer representing the number of CPU cores to be used for clustering.

    Returns:
        None. The function writes the clustered poses to a SDF file.
    '''
    # Create a directory for clustering results
    cluster_dir = Path(w_dir) / 'clustering'
    cluster_dir.mkdir(exist_ok=True)
    cluster_file = cluster_dir / f'{selection_method}_clustered.sdf'
    
    # Check if clustering has already been done for the given metric
    if not cluster_file.exists():
        # Get unique IDs from the input DataFrame
        id_list = np.unique(np.array(all_poses['ID']))
        printlog(f"*Calculating {selection_method} metrics and clustering*")
        
        # Add additional columns to the DataFrame for clustering
        all_poses['Pose_Number'] = all_poses['Pose ID'].str.split('_').str[2].astype(int)
        all_poses['Docking_program'] = all_poses['Pose ID'].str.split('_').str[1].astype(str)

        if selection_method == 'bestpose':
            # Select the best pose for each ID and docking program
            min_pose_indices = all_poses.groupby(['ID', 'Docking_program'])['Pose_Number'].idxmin()
            clustered_poses = all_poses.loc[min_pose_indices]
        elif selection_method in ['bestpose_GNINA', 'bestpose_SMINA', 'bestpose_PLANTS', 'bestpose_QVINAW', 'bestpose_QVINA2']:
            # Select the best pose for each ID based on the specified docking program
            min_pose_indices = all_poses.groupby(['ID', 'Docking_program'])['Pose_Number'].idxmin()
            clustered_poses = all_poses.loc[min_pose_indices]
            clustered_poses = clustered_poses[clustered_poses['Docking_program'] == selection_method.split('_')[1]]
        elif selection_method in CLUSTERING_METRICS.keys():
            # Perform clustering using multiple CPU cores
            clustered_dataframes = []
            
            with pebble.ProcessPool(max_workers=ncpus) as executor:
                jobs = []
                for current_id in tqdm(id_list, desc=f'Submitting {selection_method} jobs...', unit='IDs'):
                    try:
                        # Schedule the clustering job for each ID
                        job = executor.schedule(calculate_and_cluster, args=(selection_method, clustering_method, all_poses[all_poses['ID'] == current_id], protein_file), timeout=120)
                        jobs.append(job)
                    except pebble.TimeoutError as e:
                        printlog("Timeout error in pebble job creation: " + str(e))
                    except pebble.JobCancellationError as e:
                        printlog("Job cancellation error in pebble job creation: " + str(e))
                    except pebble.JobSubmissionError as e:
                        printlog("Job submission error in pebble job creation: " + str(e))
                    except Exception as e:
                        printlog("Other error in pebble job creation: " + str(e))
                for job in tqdm(jobs, total=len(id_list), desc=f'Running {selection_method} clustering...', unit='jobs'):
                    try:
                        # Get the clustering results for each job
                        res = job.result()
                        clustered_dataframes.append(res)
                    except Exception as e:
                        print(e)
                        pass
            clustered_poses = pd.concat(clustered_dataframes)
        elif selection_method in RESCORING_FUNCTIONS.keys():
            # Perform rescoring using the specified metric scoring function
            clustered_poses = rescore_docking(w_dir, protein_file, pocket_definition, software, selection_method, ncpus)
        else:
            raise ValueError(f'Invalid clustering metric: {selection_method}')
        # Clean up the Pose ID column
        clustered_poses['Pose ID'] = clustered_poses['Pose ID'].astype(str).replace('[()\',]', '', regex=True)
        # Filter the original DataFrame based on the clustered poses
        filtered_poses = all_poses[all_poses['Pose ID'].isin(clustered_poses['Pose ID'])]
        filtered_poses = filtered_poses[['Pose ID', 'Molecule', 'ID']]
        # Write the filtered poses to a SDF file
        PandasTools.WriteSDF(filtered_poses, str(cluster_file), molColName='Molecule', idName='Pose ID')
    else:
        printlog(f'Clustering using {selection_method} already done, moving to next metric...')
    return