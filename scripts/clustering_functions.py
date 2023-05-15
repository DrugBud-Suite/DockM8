from scripts.clustering_metrics import *
from scripts.utilities import *
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

def kmedoids_S_clustering(input_dataframe):
    '''This function applies kmedoids clustering to the input dataframe, which is a matrix of clustering metrics. It returns 
    the list of cluster centers and their Pose ID'''
    df = input_dataframe.copy()
    molecule_list = input_dataframe.columns.values.tolist()
    #preprocessing our data *scaling data* 
    scaler = StandardScaler()
    df[molecule_list] = scaler.fit_transform(df)
    silhouette_scores = {}
    for num_clusters in range(2,5):
        #calculating silhouette average score for every cluster and plotting them at the end
        #choosing pam method as it's more accurate
        # initialization of medoids is a greedy approach, as it's more effecient as well 
        kmedoids = KMedoids(n_clusters=num_clusters , method='pam',init='build' ,max_iter=150)
        kmedoids.fit_predict(df)
        silhouette_average_score = silhouette_score(df, kmedoids.labels_)
        silhouette_scores[num_clusters] = silhouette_average_score
    optimum_no_clusters = max(silhouette_scores, key=silhouette_scores.get)
    # # Apply optimised k-medoids clustering
    kmedoids = KMedoids(n_clusters=optimum_no_clusters, method='pam',init='build', max_iter=150)
    clusters = kmedoids.fit_predict(df)
    df['KMedoids Cluster'] = clusters
    df['Pose ID'] = molecule_list
    # Determine centers
    centroids = kmedoids.cluster_centers_
    cluster_centers = pd.DataFrame(centroids,columns = molecule_list)
    #rearranging data
    merged_df = pd.merge(df, cluster_centers, on=molecule_list, how='inner')
    merged_df = merged_df[['Pose ID']]
    #.astype(str).replace('[()\',]','', regex=False)
    return merged_df

def affinity_propagation_clustering(input_dataframe):
    '''This function applies affinity propagation clustering to the input dataframe, which is a matrix of clustering metrics. It returns 
    the list of cluster centers and their Pose ID'''
    df = input_dataframe.copy()
    molecule_list = input_dataframe.columns.values.tolist()
    #preprocessing our data *scaling data* 
    scaler = StandardScaler()
    df[molecule_list] = scaler.fit_transform(df)
    affinity_propagation = AffinityPropagation(max_iter=150)
    clusters = affinity_propagation.fit_predict(df)
    df['Affinity Cluster'] = clusters
    df['Pose ID'] = molecule_list
    # Determine centers
    centroids = affinity_propagation.cluster_centers_
    cluster_centers = pd.DataFrame(centroids,columns = molecule_list)
    #rearranging data
    merged_df = pd.merge(df, cluster_centers, on=molecule_list, how='inner')
    merged_df = merged_df[['Pose ID']]
    #.astype(str).replace('[()\',]','', regex=False)
    return merged_df

def metric_calculation_failure_handling(x, y, metric, protein_file):
    metrics = {'RMSD': simpleRMSD_calc, 'spyRMSD': spyRMSD_calc, 'espsim': espsim_calc, 'USRCAT': USRCAT_calc, 'SPLIF': SPLIF_calc, '3DScore': '3DScore', 'bestpose': 'bestpose', 'symmRMSD': symmRMSD_calc}
    if metric == 'spyRMSD':
        try:
            return metrics[metric](x, y, protein_file)
        except Exception as e:
            return metrics['RMSD'](x, y, protein_file)
    else:
        try:
            return metrics[metric](x, y, protein_file)
        except Exception as e:
            printlog(f'Failed to calculate {metric} and cluster : {e}')
            return 0

def matrix_calculation_and_clustering(metric, method, df, protein_file):
    # print("Starting matrix_calculation_and_clustering function")
    methods = {'KMedoids': kmedoids_S_clustering, 'AffProp': affinity_propagation_clustering}
    
    # print("Creating subsets and indices")
    subsets = np.array(list(itertools.combinations(df['Molecule'], 2)))
    indices = {mol: idx for idx, mol in enumerate(df['Molecule'].values)}
    
    # print("Vectorizing metric_calculation_failure_handling")
    vectorized_calc_vec = np.vectorize(metric_calculation_failure_handling)
    
    # print("Calculating results using vectorized_calc_vec")
    results = vectorized_calc_vec(subsets[:,0], subsets[:,1], metric if metric != '3DScore' else 'spyRMSD', protein_file)
    
    # print("Assigning i and j values")
    i, j = np.array([indices[x] for x in subsets[:,0]]), np.array([indices[y] for y in subsets[:,1]])
    
    # print("Creating and filling matrix")
    matrix = np.zeros((len(df), len(df)))
    matrix[i, j] = results
    matrix[j, i] = results
    
    if metric == '3DScore':
        # print("Entering 3DScore section")
        output_df = pd.DataFrame(matrix, index=df['Pose ID'].values.tolist(), columns=df['Pose ID'].values.tolist())
        output_df['3DScore'] = output_df.sum(axis=1)
        output_df.sort_values(by='3DScore', ascending=True, inplace=True)
        output_df = output_df.head(1)
        output_df = pd.DataFrame(output_df.index, columns=['Pose ID'])
        output_df['Pose ID'] = output_df['Pose ID'].astype(str).str.replace('[()\',]','', regex=False)
        # print("Exiting 3DScore section")
        return output_df
    else:
        # print("Entering non-3DScore section")
        matrix_df = pd.DataFrame(matrix, index=df['Pose ID'].values.tolist(), columns=df['Pose ID'].values.tolist())
        matrix_df.fillna(0)
        clust_df = methods[method](matrix_df)
        # print("Exiting non-3DScore section")
        return clust_df


def cluster(metric, method, w_dir, protein_file, all_poses, ncpus):
    '''This function clusters all poses according to the metric selected using multiple CPU cores'''
    create_temp_folder(w_dir+'/temp/clustering/')
    if os.path.isfile(w_dir + '/temp/clustering/' + metric + '_clustered.sdf') == False:
        id_list = np.unique(np.array(all_poses['ID']))
        printlog(f"*Calculating {metric} metrics and clustering*")
        best_pose_filters = {'bestpose': ('_1', '_01'),
                            'bestpose_GNINA': ('GNINA_1','GNINA_01'),
                            'bestpose_SMINA': ('SMINA_1','SMINA_01'),
                            'bestpose_PLANTS': ('PLANTS_1','PLANTS_01')}
        if metric in best_pose_filters:
            filter = best_pose_filters[metric]
            clustered_poses = all_poses[all_poses['Pose ID'].str.endswith(filter)]
            clustered_poses = clustered_poses[['Pose ID']]
            clustered_poses['Pose ID'] = clustered_poses['Pose ID'].astype(str).str.replace('[()\',]','', regex=False)
        else:
            if ncpus > 1:
                clustered_dataframes = []
                with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                    printlog('Submitting parallel jobs...')
                    tic = time.perf_counter()
                    jobs = []
                    for current_id in tqdm(id_list, desc='Submitting parallel jobs...', unit='IDs'):
                        try:
                            job = executor.submit(matrix_calculation_and_clustering, metric, method, all_poses[all_poses['ID']==current_id], protein_file)
                            jobs.append(job)
                        except Exception as e:
                            printlog("Error in concurrent futures job creation: "+ str(e))	
                    toc = time.perf_counter()
                    printlog(f'Finished submitting jobs in {toc-tic:0.4f}, now running jobs...')
                    for job in tqdm(concurrent.futures.as_completed(jobs), total=len(id_list), desc='Running clustering jobs...', unit='jobs'):
                        try:
                            res = job.result(timeout=60)
                            clustered_dataframes.append(res)
                        except Exception as e:
                            printlog("Error in concurrent futures job run: "+ str(e))
                clustered_poses = pd.concat(clustered_dataframes)
            else:
                clustered_poses = matrix_calculation_and_clustering(metric, method, all_poses, id_list, protein_file)
        clustered_poses['Pose ID'] = clustered_poses['Pose ID'].astype(str).replace('[()\',]','', regex=True)
        clustered_poses = pd.merge(all_poses, clustered_poses, on='Pose ID')
        clustered_poses = clustered_poses[['Pose ID', 'Molecule', 'ID']]
        save_path = w_dir + '/temp/clustering/' + metric + '_clustered.sdf'
        PandasTools.WriteSDF(clustered_poses, save_path, molColName='Molecule', idName='Pose ID')
    else:
        printlog(f'Clustering using {metric} already done, moving to next metric...')
    return

import pebble
import traceback

def cluster_pebble(metric, method, w_dir, protein_file, all_poses, ncpus):
    '''This function clusters all poses according to the metric selected using multiple CPU cores'''
    create_temp_folder(w_dir+'/temp/clustering/')
    if os.path.isfile(w_dir + '/temp/clustering/' + metric + '_clustered.sdf') == False:
        id_list = np.unique(np.array(all_poses['ID']))
        printlog(f"*Calculating {metric} metrics and clustering*")
        best_pose_filters = {'bestpose': ('_1', '_01'),
                            'bestpose_GNINA': ('GNINA_1','GNINA_01'),
                            'bestpose_SMINA': ('SMINA_1','SMINA_01'),
                            'bestpose_PLANTS': ('PLANTS_1','PLANTS_01')}
        if metric in best_pose_filters:
            filter = best_pose_filters[metric]
            clustered_poses = all_poses[all_poses['Pose ID'].str.endswith(filter)]
            clustered_poses = clustered_poses[['Pose ID']]
            clustered_poses['Pose ID'] = clustered_poses['Pose ID'].astype(str).str.replace('[()\',]','', regex=False)
        if metric == 'bestpose_PLANTS':
            grouped_poses = clustered_poses.groupby('ID')
            min_chemplp_rows = grouped_poses.apply(lambda x: x.loc[x['CHEMPLP'].idxmin()])
            clustered_poses = min_chemplp_rows
        if metric == 'bestpose_GNINA':
            grouped_poses = clustered_poses.groupby('ID')
            min_gnina_rows = grouped_poses.apply(lambda x: x.loc[x['CNNscore'].idxmin()])
            clustered_poses = min_gnina_rows
        if metric == 'bestpose_SMINA':
            grouped_poses = clustered_poses.groupby('ID')
            min_smina_rows = grouped_poses.apply(lambda x: x.loc[x['SMINA_Affinity'].idxmin()])
            clustered_poses = min_smina_rows
        if metric == 'bestpose':
            grouped_poses = clustered_poses.groupby('ID')
            min_chemplp_rows = grouped_poses.apply(lambda x: x.loc[x['CHEMPLP'].idxmin()])
            min_gnina_rows = grouped_poses.apply(lambda x: x.loc[x['CNNscore'].idxmin()])
            min_smina_rows = grouped_poses.apply(lambda x: x.loc[x['SMINA_Affinity'].idxmin()])
            clustered_poses = pd.concat([min_smina_rows, min_chemplp_rows, min_gnina_rows])
        else:
            if ncpus > 1:
                clustered_dataframes = []
                with pebble.ProcessPool(max_workers=ncpus) as executor:
                    printlog('Submitting parallel jobs...')
                    tic = time.perf_counter()
                    jobs = []
                    for current_id in tqdm(id_list, desc='Submitting parallel jobs...', unit='IDs'):
                        try:
                            job = executor.schedule(matrix_calculation_and_clustering, args=(metric, method, all_poses[all_poses['ID']==current_id], protein_file), timeout=120)
                            jobs.append(job)
                        except Exception as e:
                            printlog("Error in pebble job creation: "+ str(e))	
                    toc = time.perf_counter()
                    printlog(f'Finished submitting jobs in {toc-tic:0.4f}, now running jobs...')
                    for job in tqdm(jobs, total=len(id_list), desc='Running clustering jobs...', unit='jobs'):
                        try:
                            res = job.result()
                            clustered_dataframes.append(res)
                        except Exception as e:
                            pass
                clustered_poses = pd.concat(clustered_dataframes)
            else:
                clustered_poses = matrix_calculation_and_clustering(metric, method, all_poses, id_list, protein_file)
        clustered_poses['Pose ID'] = clustered_poses['Pose ID'].astype(str).replace('[()\',]','', regex=True)
        clustered_poses = pd.merge(all_poses, clustered_poses, on='Pose ID')
        clustered_poses = clustered_poses[['Pose ID', 'Molecule', 'ID']]
        save_path = w_dir + '/temp/clustering/' + metric + '_clustered.sdf'
        PandasTools.WriteSDF(clustered_poses, save_path, molColName='Molecule', idName='Pose ID')
    else:
        printlog(f'Clustering using {metric} already done, moving to next metric...')
    return
