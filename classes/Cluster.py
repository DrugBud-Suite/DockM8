from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import concurrent.futures

class Cluster:
    def __init__(self, clustering_method, working_directory, protein_file, all_poses, number_cpus):
        self.clustering_method = clustering_method
        self.protein_file = protein_file
        self.working_directory = working_directory
        self.all_poses = all_poses
        self.number_cpus = number_cpus
        self.metrics = {'RMSD': simpleRMSD_calc, 'spyRMSD': spyRMSD_calc, 'espsim': espsim_calc, 'USRCAT': USRCAT_calc, 'SPLIF': SPLIF_calc, '3DScore': '3DScore', 'bestpose': 'bestpose', 'symmRMSD': symmRMSD_calc}
        self.methods = {'KMedoids': kmedoids_S_clustering, 'AffProp': affinity_propagation_clustering}
    def cluster(self, metric):
        if metric not in self.metrics:
            return 0
        if os.path.isfile(w_dir+f'/temp/clustering/{metric}_clustered.sdf'):
            return 0
        create_temp_folder(w_dir+'/temp/clustering/')
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
        
    def matrix_calculation_and_clustering_futures_failure_handling(self, metric, method, df):
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


def cluster(self, metric):
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
            with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
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