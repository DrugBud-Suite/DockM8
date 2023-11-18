from typing import Dict, Any, Callable, Union
import pebble
import traceback
from scripts.clustering_metrics import *
from scripts.utilities import *
from scripts.rescoring_functions import *
import pandas as pd
import numpy as np
import math
import os
import functools
from tqdm import tqdm
from sklearn.cluster import AffinityPropagation
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import itertools
from rdkit.Chem import PandasTools
from IPython.display import display
import multiprocessing
import concurrent.futures
import time
from pathlib import Path



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
        print(df)
        molecule_list = input_dataframe.columns.tolist()

        # Scale the values of the molecules in the dataframe using StandardScaler
        scaler = StandardScaler()
        df[molecule_list] = scaler.fit_transform(df)

        silhouette_scores = {}
        for num_clusters in range(2, 5):
            # Calculate silhouette average score for every cluster and select the optimal number of clusters
            # Choosing PAM method as it's more accurate
            kmedoids = KMedoids(
                n_clusters=num_clusters,
                method='pam',
                init='build',
                max_iter=150
            )
            kmedoids.fit_predict(df)
            silhouette_average_score = silhouette_score(df, kmedoids.labels_)
            silhouette_scores[num_clusters] = silhouette_average_score

        optimum_no_clusters = max(silhouette_scores, key=silhouette_scores.get)

        # Apply optimized k-medoids clustering
        kmedoids = KMedoids(
            n_clusters=optimum_no_clusters,
            method='pam',
            init='build',
            max_iter=150
        )
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
    clustering_metrics: Dict[str, Union[str, Callable]] = {'RMSD': simpleRMSD_calc,
                                                'spyRMSD': spyRMSD_calc,
                                                'espsim': espsim_calc,
                                                'USRCAT': USRCAT_calc,
                                                'SPLIF': SPLIF_calc,
                                                '3DScore': '3DScore'}

    clustering_methods: Dict[str, Callable] = {'KMedoids': kmedoids_S_clustering,
                                    'AffProp': affinity_propagation_clustering}

    subsets = np.array(list(itertools.combinations(df['Molecule'], 2)))
    indices = {mol: idx for idx, mol in enumerate(df['Molecule'].values)}

    if clustering_metric == '3DScore':
        metric_func = clustering_metrics['spyRMSD']
    elif clustering_metric in clustering_metrics:
        metric_func = clustering_metrics[clustering_metric]
    else:
        raise ValueError(f"Invalid metric '{clustering_metric}'")

    vectorized_calc_vec = np.vectorize(lambda x, y: metric_func(x, y, protein_file))

    results = vectorized_calc_vec(subsets[:, 0], subsets[:, 1])

    i = np.array([indices[x] for x in subsets[:, 0]])
    j = np.array([indices[y] for y in subsets[:, 1]])

    matrix = np.zeros((len(df), len(df)))
    matrix[i, j] = results
    matrix[j, i] = results

    if clustering_metric == '3DScore':
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
        matrix_df = pd.DataFrame(matrix,
                                 index=df['Pose ID'].values.tolist(),
                                 columns=df['Pose ID'].values.tolist())
        matrix_df.fillna(0)
        clustered_df = clustering_methods[clustering_method](matrix_df)
        return clustered_df


def cluster_pebble(clustering_metric : str, clustering_method : str, w_dir : Path, protein_file: Path, pocket_definition: dict, software: Path, all_poses : pd.DataFrame, ncpus : int):
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
    cluster_dir = Path(w_dir) / 'clustering'
    cluster_dir.mkdir(exist_ok=True)
    cluster_file = cluster_dir / f'{clustering_metric}_clustered.sdf'
    if not cluster_file.exists():
        id_list = np.unique(np.array(all_poses['ID']))
        printlog(f"*Calculating {clustering_metric} metrics and clustering*")
        
        all_poses['Pose_Number'] = all_poses['Pose ID'].str.split('_').str[2].astype(int)
        all_poses['Docking_program'] = all_poses['Pose ID'].str.split('_').str[1].astype(str)

        if clustering_metric == 'bestpose':
            min_pose_indices = all_poses.groupby(['ID', 'Docking_program'])['Pose_Number'].idxmin()
            clustered_poses = all_poses.loc[min_pose_indices]
        elif clustering_metric in ['bestpose_GNINA', 'bestpose_SMINA', 'bestpose_PLANTS', 'bestpose_QVINAW', 'bestpose_QVINA2']:
            min_pose_indices = all_poses.groupby('ID')['Pose_Number'].idxmin()
            docking_program = clustering_metric.split('_')[1]
            clustered_poses = all_poses[all_poses['Docking_program'] == docking_program]
        elif clustering_metric in ['RMSD', 'spyRMSD', 'espsim', 'USRCAT', '3DScore']:
            clustered_dataframes = []
            with pebble.ProcessPool(max_workers=ncpus) as executor:
                tic = time.perf_counter()
                jobs = []
                for current_id in tqdm(id_list, desc=f'Submitting {clustering_metric} jobs...', unit='IDs'):
                    try:
                        job = executor.schedule(calculate_and_cluster, args=(clustering_metric, clustering_method, all_poses[all_poses['ID'] == current_id], protein_file), timeout=120)
                        jobs.append(job)
                    except pebble.TimeoutError as e:
                        printlog("Timeout error in pebble job creation: " + str(e))
                    except pebble.JobCancellationError as e:
                        printlog("Job cancellation error in pebble job creation: " + str(e))
                    except pebble.JobSubmissionError as e:
                        printlog("Job submission error in pebble job creation: " + str(e))
                    except Exception as e:
                        printlog("Other error in pebble job creation: " + str(e))
                toc = time.perf_counter()
                for job in tqdm(jobs, total=len(id_list), desc=f'Running {clustering_metric} clustering...', unit='jobs'):
                    try:
                        res = job.result()
                        clustered_dataframes.append(res)
                    except Exception as e:
                        print(e)
                        pass
            clustered_poses = pd.concat(clustered_dataframes)
        else:
            clustered_poses = rescore_docking(w_dir, protein_file, pocket_definition, software, clustering_metric, ncpus)
        clustered_poses['Pose ID'] = clustered_poses['Pose ID'].astype(str).replace('[()\',]', '', regex=True)
        filtered_poses = all_poses[all_poses['Pose ID'].isin(clustered_poses['Pose ID'])]
        filtered_poses = filtered_poses[['Pose ID', 'Molecule', 'ID']]
        PandasTools.WriteSDF(filtered_poses, str(cluster_file), molColName='Molecule', idName='Pose ID')
    else:
        printlog(f'Clustering using {clustering_metric} already done, moving to next metric...')
    return filtered_poses