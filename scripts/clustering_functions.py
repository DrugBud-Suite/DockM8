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

def cluster(metric, method, w_dir, protein_file, all_poses):
    create_temp_folder(w_dir+'/temp/clustering/')
    def matrix_calculation_and_clustering(metric, method, df, id_list, protein_file): 
        clustered_dataframes = []
        print("*Calculating {} metrics and clustering*".format(metric))
        metrics = {'RMSD': simpleRMSD_calc, 'spyRMSD': spyRMSD_calc, 'espsim': espsim_calc, 'USRCAT': USRCAT_calc, 'SPLIF': SPLIF_calc, '3DScore': '3DScore', 'bestpose': 'bestpose', 'symmRMSD': symmRMSD_calc}
        methods = {'KMedoids': kmedoids_S_clustering, 'AffProp': affinity_propagation_clustering}
        for id in tqdm(id_list, desc='Metric calculation and clustering', unit='IDs'):
            if metric == '3DScore':
                df_filtered = df[df['ID']==id]
                subsets = np.array(list(itertools.combinations(df_filtered['Molecule'], 2)))
                indices = {mol: idx for idx, mol in enumerate(df_filtered['Molecule'].values)}
                results = np.zeros(len(subsets))
                for k, (x, y) in enumerate(subsets):
                    try:
                        results[k] = (metrics['spyRMSD'](x, y, protein_file))
                    except:
                        results[k] = (metrics['RMSD'](x, y, protein_file))
                i, j = np.array([indices[x] for x in subsets[:,0]]), np.array([indices[y] for y in subsets[:,1]])
                matrix = np.zeros((len(df_filtered), len(df_filtered)))
                matrix[i, j] = results
                matrix[j, i] = results
                matrix_df = pd.DataFrame(matrix, index=df_filtered['Pose ID'].values.tolist(), columns=df_filtered['Pose ID'].values.tolist())
                matrix_df['3DScore'] = matrix_df.sum(axis=1)
                matrix_df.sort_values(by='3DScore', ascending=True, inplace=True)
                matrix_df = matrix_df.head(1)
                matrix_df = pd.DataFrame(matrix_df.index, columns=['Pose ID'])
                matrix_df['Pose ID'] = matrix_df['Pose ID'].astype(str).str.replace('[()\',]','', regex=False)
                clustered_dataframes.append(matrix_df)
            else:
                try:
                    df_filtered = df[df['ID']==id]
                    subsets = np.array(list(itertools.combinations(df_filtered['Molecule'], 2)))
                    indices = {mol: idx for idx, mol in enumerate(df_filtered['Molecule'].values)}
                    results = np.array([metrics[metric](x, y, protein_file) for x, y in subsets])
                    i, j = np.array([indices[x] for x in subsets[:,0]]), np.array([indices[y] for y in subsets[:,1]])
                    matrix = np.zeros((len(df_filtered), len(df_filtered)))
                    matrix[i, j] = results
                    matrix[j, i] = results
                    matrix_df = pd.DataFrame(matrix, index=df_filtered['Pose ID'].values.tolist(), columns=df_filtered['Pose ID'].values.tolist())
                    clust_df = methods[method](matrix_df)
                    clustered_dataframes.append(clust_df)
                except Exception as e:
                    print(f'Failed to calculate metrics and cluster ID: {id} due to : {e}')
        clustered_poses = pd.concat(clustered_dataframes)
        clustered_poses['Pose ID'] = clustered_poses['Pose ID'].astype(str).replace('[()\',]','', regex=True)
        return clustered_poses
    if os.path.isfile(w_dir + '/temp/clustering/' + metric + '_clustered.sdf') == False:
        id_list = np.unique(np.array(all_poses['ID']))
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
            clustered_poses = matrix_calculation_and_clustering(metric, method, all_poses, id_list, protein_file)
        clustered_poses = pd.merge(all_poses, clustered_poses, on='Pose ID')
        clustered_poses = clustered_poses[['Pose ID', 'Molecule', 'ID']]
        save_path = w_dir + '/temp/clustering/' + metric + '_clustered.sdf'
        PandasTools.WriteSDF(clustered_poses, save_path, molColName='Molecule', idName='Pose ID')
    else:
        print(f'Clustering using {metric} already done, moving to next metric...')
    return

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
            print(f'Failed to calculate {metric} and cluster : {e}')
            return 0

def matrix_calculation_and_clustering_futures_failure_handling(metric, method, df, protein_file):
    methods = {'KMedoids': kmedoids_S_clustering, 'AffProp': affinity_propagation_clustering}
    if metric == '3DScore':
        subsets = np.array(list(itertools.combinations(df['Molecule'], 2)))
        indices = {mol: idx for idx, mol in enumerate(df['Molecule'].values)}
        vectorized_calc_vec = np.vectorize(metric_calculation_failure_handling)
        results = vectorized_calc_vec(subsets[:,0], subsets[:,1], 'spyRMSD', protein_file)
        i, j = np.array([indices[x] for x in subsets[:,0]]), np.array([indices[y] for y in subsets[:,1]])
        matrix = np.zeros((len(df), len(df)))
        matrix[i, j] = results
        matrix[j, i] = results
        output_df = pd.DataFrame(matrix, index=df['Pose ID'].values.tolist(), columns=df['Pose ID'].values.tolist())
        output_df['3DScore'] = output_df.sum(axis=1)
        output_df.sort_values(by='3DScore', ascending=True, inplace=True)
        output_df = output_df.head(1)
        output_df = pd.DataFrame(output_df.index, columns=['Pose ID'])
        output_df['Pose ID'] = output_df['Pose ID'].astype(str).str.replace('[()\',]','', regex=False)
        return output_df
    else:
        subsets = np.array(list(itertools.combinations(df['Molecule'], 2)))
        indices = {mol: idx for idx, mol in enumerate(df['Molecule'].values)}
        vectorized_calc_vec = np.vectorize(metric_calculation_failure_handling)
        results = vectorized_calc_vec(subsets[:,0], subsets[:,1], metric, protein_file)
        i, j = np.array([indices[x] for x in subsets[:,0]]), np.array([indices[y] for y in subsets[:,1]])
        matrix = np.zeros((len(df), len(df)))
        matrix[i, j] = results
        matrix[j, i] = results
        matrix_df = pd.DataFrame(matrix, index=df['Pose ID'].values.tolist(), columns=df['Pose ID'].values.tolist())
        matrix_df.fillna(0)
        clust_df = methods[method](matrix_df)
        return clust_df

def cluster_futures(metric, method, w_dir, protein_file, all_poses):
    create_temp_folder(w_dir+'/temp/clustering/')
    if os.path.isfile(w_dir + '/temp/clustering/' + metric + '_clustered.sdf') == False:
        id_list = np.unique(np.array(all_poses['ID']))
        print(f"*Calculating {metric} metrics and clustering*")
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
            clustered_dataframes = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=int(multiprocessing.cpu_count()/2)) as executor:
                print('Submitting parallel jobs...')
                tic = time.perf_counter()
                jobs = []
                for current_id in tqdm(id_list, desc='Submitting parallel jobs...', unit='IDs'):
                    try:
                        job = executor.submit(matrix_calculation_and_clustering_futures_failure_handling, metric, method, all_poses[all_poses['ID']==current_id], protein_file)
                        jobs.append(job)
                    except Exception as e:
                        print("Error in concurrent futures job creation: ", str(e))	
                toc = time.perf_counter()
                print(f'Finished submitting jobs in {toc-tic:0.4f}, now running jobs...')
                for job in tqdm(concurrent.futures.as_completed(jobs), total=len(id_list), desc='Running clustering jobs...', unit='jobs'):
                    try:
                        res = job.result()
                        clustered_dataframes.append(res)
                    except Exception as e:
                        print("Error in concurrent futures job run: ", str(e))
            clustered_poses = pd.concat(clustered_dataframes)
        clustered_poses['Pose ID'] = clustered_poses['Pose ID'].astype(str).replace('[()\',]','', regex=True)
        clustered_poses = pd.merge(all_poses, clustered_poses, on='Pose ID')
        clustered_poses = clustered_poses[['Pose ID', 'Molecule', 'ID']]
        save_path = w_dir + '/temp/clustering/' + metric + '_clustered.sdf'
        PandasTools.WriteSDF(clustered_poses, save_path, molColName='Molecule', idName='Pose ID')
    else:
        print(f'Clustering using {metric} already done, moving to next metric...')
    return