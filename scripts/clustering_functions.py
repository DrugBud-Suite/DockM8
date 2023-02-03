import pandas as pd
import numpy as np
import math
import os
import functools
from tqdm import tqdm
from spyrmsd import io, rmsd
from espsim import GetEspSim
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import itertools
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import PandasTools
from IPython.display import display
from spyrmsd import molecule
from spyrmsd.optional import rdkit as rdkit_loader
import oddt
import oddt.shape
import oddt.fingerprints
import oddt.toolkits.rdk
import multiprocessing
import dask.dataframe as dd
from dask import delayed
import dask
import concurrent.futures
import progressbar

def create_clustering_folder(path):
    if os.path.isdir(path) == True:
        print('Clustering folder already exists')
    else:
        os.mkdir(path)
        print('Clustering folder was created')
        
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

def simpleRMSD_calc(*args):
# MCS identification between reference pose and target pose
    r=rdFMCS.FindMCS([args[0],args[1]])
# Atom map for reference and target              
    a=args[0].GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
    b=args[1].GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
# Atom map generation     
    amap=list(zip(a,b))
# distance calculation per atom pair
    distances=[]
    for atomA, atomB in amap:
        pos_A=args[0].GetConformer().GetAtomPosition (atomA)
        pos_B=args[1].GetConformer().GetAtomPosition (atomB)
        coord_A=np.array((pos_A.x,pos_A.y,pos_A.z))
        coord_B=np.array ((pos_B.x,pos_B.y,pos_B.z))
        dist_numpy = np.linalg.norm(coord_A-coord_B)        
        distances.append(dist_numpy)      
# This is the RMSD formula from wikipedia
    rmsd=math.sqrt(1/len(distances)*sum([i*i for i in distances])) 
    return round(rmsd, 3)

def spyRMSD_calc(*args):
    mol = args[0][0] if type(args[0]) == tuple else args[0]
    jmol = args[0][1] if type(args[0]) == tuple else args[1]
    spyrmsd_mol = molecule.Molecule.from_rdkit(mol)
    spyrmsd_jmol = molecule.Molecule.from_rdkit(jmol)
    spyrmsd_mol.strip()
    spyrmsd_jmol.strip()
    coords_ref = spyrmsd_mol.coordinates
    anum_ref = spyrmsd_mol.atomicnums
    adj_ref = spyrmsd_mol.adjacency_matrix
    coords_test = spyrmsd_jmol.coordinates
    anum_test = spyrmsd_jmol.atomicnums
    adj_test = spyrmsd_jmol.adjacency_matrix
    spyRMSD = rmsd.symmrmsd(coords_ref,coords_test,anum_ref,anum_test,adj_ref,adj_test)
    return round(spyRMSD, 3)

def espsim_calc(*args):
    return GetEspSim(args[0], args[1])

def SPLIF_calc(*args):
    pocket_file = args[2].replace('.pdb', '_pocket.pdb')
    protein=next(oddt.toolkit.readfile('pdb', pocket_file))
    protein.protein = True
    splif_mol = oddt.toolkits.rdk.Molecule(args[0])
    splif_jmol = oddt.toolkits.rdk.Molecule(args[1])
    mol_fp=oddt.fingerprints.SimpleInteractionFingerprint(splif_mol, protein)
    jmol_fp=oddt.fingerprints.SimpleInteractionFingerprint(splif_jmol, protein)
    SPLIF_sim = oddt.fingerprints.tanimoto(mol_fp, jmol_fp)
    return round(SPLIF_sim, 3)

def USRCAT_calc(*args):
    shape_mol = oddt.toolkits.rdk.Molecule(args[0])
    shape_jmol = oddt.toolkits.rdk.Molecule(args[1])
    mol_fp=oddt.shape.usr_cat(shape_mol)
    jmol_fp=oddt.shape.usr_cat(shape_jmol)
    usr_sim = oddt.shape.usr_similarity(mol_fp, jmol_fp)
    return round(usr_sim, 3)

#NOT WORKING!
def symmRMSD_calc(*args):
    display(args[0], args[1])
    rms = Chem.rdMolAlign.CalcRMS(args[0], args[1])
    return round(rms, 3)

def cluster(method, w_dir, protein_file):
    def matrix_calculation_and_clustering(method, df, id_list, protein_file, w_dir): 
        clustered_dataframes = []
        print("*Calculating {} metrics and clustering*".format(method))
        methods = {'RMSD': simpleRMSD_calc, 'spyRMSD': spyRMSD_calc, 'espsim': espsim_calc, 'USRCAT': USRCAT_calc, 'SPLIF': SPLIF_calc, '3DScore': '3DScore', 'bestpose': 'bestpose', 'symmRMSD': symmRMSD_calc}
        for id in tqdm(id_list):
            if method == 'bestpose':
                df_name = df[df['ID']==id]
                df_name[['CHEMPLP', 'SMINA_Affinity', 'CNNaffinity']] = df_name[['CHEMPLP', 'SMINA_Affinity', 'CNNaffinity']].apply(pd.to_numeric, errors='coerce')
                best_row_CHEMPLP = df_name.loc[df_name.groupby(['ID'])['CHEMPLP'].idxmin()]
                best_row_SMINA = df_name.loc[df_name.groupby(['ID'])['SMINA_Affinity'].idxmin()]
                best_row_GNINA = df_name.loc[df_name.groupby(['ID'])['CNNaffinity'].idxmax()]
                table = pd.concat([best_row_GNINA, best_row_SMINA, best_row_CHEMPLP])
                table.reset_index(inplace=True)
                table = pd.DataFrame(table['Pose ID'])
                table['Pose ID'] = table['Pose ID'].astype(str).str.replace('[()\',]','', regex=False)
                clustered_dataframes.append(table)
            elif method == '3DScore':
                df_name = df[df['ID']==id]
                df_name.index = range(len(df_name['Molecule']))
                table = pd.DataFrame(0.0, index=[df_name['Pose ID']], columns=df_name['Molecule'])
                for subset in itertools.combinations(df_name['Molecule'], 2):
                    try:
                        result = methods['spyRMSD'](subset[0], subset[1])
                    except:
                        result = methods['RMSD'](subset[0], subset[1])
                    table.iloc[df_name[df_name['Molecule']==subset[0]].index.values, df_name[df_name['Molecule']==subset[1]].index.values] = 0 if np.isnan(result) else result
                    table.iloc[df_name[df_name['Molecule']==subset[1]].index.values, df_name[df_name['Molecule']==subset[0]].index.values] = 0 if np.isnan(result) else result
                table['3DScore'] = table.sum(axis=1)
                table.sort_values(by='3DScore', ascending=True)
                table = table.head(1)
                table.reset_index(inplace=True)
                table = pd.DataFrame(table['Pose ID'])
                table['Pose ID'] = table['Pose ID'].astype(str).str.replace('[()\',]','', regex=False)
                clustered_dataframes.append(table)
            else:
                try:
                    df_name = df[df['ID']==id]
                    df_name.index = range(len(df_name['Molecule']))
                    table = pd.DataFrame(0.0, index=[df_name['Pose ID']], columns=df_name['Molecule'])
                    for subset in itertools.combinations(df_name['Molecule'], 2):
                        result = methods[method](subset[0], subset[1], protein_file)
                        table.iloc[df_name[df_name['Molecule']==subset[0]].index.values, df_name[df_name['Molecule']==subset[1]].index.values] = 0 if np.isnan(result) else result
                        table.iloc[df_name[df_name['Molecule']==subset[1]].index.values, df_name[df_name['Molecule']==subset[0]].index.values] = 0 if np.isnan(result) else result
                    clust_df = kmedoids_S_clustering(table)
                    clust_df=clust_df['Pose ID']
                    clustered_dataframes.append(clust_df)
                except:
                    print(f'Failed to calculate metrics and cluster ID: {id}')
        full_df = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Pose ID'], how='outer'), clustered_dataframes)
        full_df['Pose ID'] = full_df['Pose ID'].astype(str).replace('[()\',]','', regex=True)
        return full_df
    print('Loading all poses SDF file...')
    all_poses = PandasTools.LoadSDF(w_dir+'/temp/allposes.sdf', idName='Pose ID', molColName='Molecule', includeFingerprints=False, strictParsing=True)
    print('Finished loading all poses SDF file...')
    id_list = np.unique(np.array(all_poses['ID']))
    create_clustering_folder(w_dir+'/temp/clustering/')
    clustered_poses = matrix_calculation_and_clustering(method, all_poses, id_list, protein_file, w_dir)
    clustered_poses = pd.merge(all_poses, clustered_poses, on='Pose ID')
    # keep only the necessary columns
    clustered_poses = clustered_poses[['Pose ID', 'Molecule', 'ID']]
    save_path = w_dir + '/temp/clustering/' + method + '_clustered.sdf'
    PandasTools.WriteSDF(clustered_poses, save_path, molColName='Molecule', idName='Pose ID')
    return

def cluster_numpy(method, w_dir, protein_file):
    def matrix_calculation_and_clustering(method, df, id_list, protein_file): 
        clustered_dataframes = []
        print("*Calculating {} metrics and clustering*".format(method))
        methods = {'RMSD': simpleRMSD_calc, 'spyRMSD': spyRMSD_calc, 'espsim': espsim_calc, 'USRCAT': USRCAT_calc, 'SPLIF': SPLIF_calc, '3DScore': '3DScore', 'bestpose': 'bestpose', 'symmRMSD': symmRMSD_calc}
        for id in tqdm(id_list):
            if method == 'bestpose':
                df_filtered = df[df['ID']==id]
                best_pose_output = df_filtered[df_filtered['Pose ID'].str.endswith(('_1', '_01'))]
                best_pose_output = best_pose_output[['Pose ID']]
                best_pose_output['Pose ID'] = best_pose_output['Pose ID'].astype(str).str.replace('[()\',]','', regex=False)
                clustered_dataframes.append(best_pose_output)
            elif method == '3DScore':
                df_filtered = df[df['ID']==id]
                subsets = np.array(list(itertools.combinations(df_filtered['Molecule'], 2)))
                indices = {mol: idx for idx, mol in enumerate(df_filtered['Molecule'].values)}
                results = np.zeros(len(subsets))
                for k, (x, y) in enumerate(subsets):
                    try:
                        results[k] = (methods['spyRMSD'](x, y, protein_file))
                    except:
                        results[k] = (methods['RMSD'](x, y, protein_file))
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
                    results = np.array([methods[method](x, y, protein_file) for x, y in subsets])
                    i, j = np.array([indices[x] for x in subsets[:,0]]), np.array([indices[y] for y in subsets[:,1]])
                    matrix = np.zeros((len(df_filtered), len(df_filtered)))
                    matrix[i, j] = results
                    matrix[j, i] = results
                    matrix_df = pd.DataFrame(matrix, index=df_filtered['Pose ID'].values.tolist(), columns=df_filtered['Pose ID'].values.tolist())
                    clust_df = kmedoids_S_clustering(matrix_df)
                    clustered_dataframes.append(clust_df)
                except Exception as e:
                    print(f'Failed to calculate metrics and cluster ID: {id} due to : {e}')
        clustered_poses = pd.concat(clustered_dataframes)
        clustered_poses['Pose ID'] = clustered_poses['Pose ID'].astype(str).replace('[()\',]','', regex=True)
        return clustered_poses
    print('Loading all poses SDF file...')
    all_poses = PandasTools.LoadSDF(w_dir+'/temp/allposes.sdf', idName='Pose ID', molColName='Molecule', includeFingerprints=False, strictParsing=True)
    print('Finished loading all poses SDF file...')
    id_list = np.unique(np.array(all_poses['ID']))
    create_clustering_folder(w_dir+'/temp/clustering/')
    clustered_poses = matrix_calculation_and_clustering(method, all_poses, id_list, protein_file)
    clustered_poses = pd.merge(all_poses, clustered_poses, on='Pose ID')
    # keep only the necessary columns
    clustered_poses = clustered_poses[['Pose ID', 'Molecule', 'ID']]
    save_path = w_dir + '/temp/clustering/' + method + '_clustered.sdf'
    PandasTools.WriteSDF(clustered_poses, save_path, molColName='Molecule', idName='Pose ID')
    return

def matrix_calculation_and_clustering_futures(method, df, protein_file):
    methods = {'RMSD': simpleRMSD_calc, 'spyRMSD': spyRMSD_calc, 'espsim': espsim_calc, 'USRCAT': USRCAT_calc, 'SPLIF': SPLIF_calc, '3DScore': '3DScore', 'bestpose': 'bestpose', 'symmRMSD': symmRMSD_calc}
    if method == 'bestpose':
        best_pose_output = df[df['Pose ID'].str.endswith(('_1', '_01'))]
        best_pose_output = best_pose_output[['Pose ID']]
        best_pose_output['Pose ID'] = best_pose_output['Pose ID'].astype(str).str.replace('[()\',]','', regex=False)
        return best_pose_output
    elif method == '3DScore':
        subsets = np.array(list(itertools.combinations(df['Molecule'], 2)))
        indices = {mol: idx for idx, mol in enumerate(df['Molecule'].values)}
        results = np.zeros(len(subsets))
        for k, (x, y) in enumerate(subsets):
            try:
                results[k] = (methods['spyRMSD'](x, y, protein_file))
            except:
                results[k] = (methods['RMSD'](x, y, protein_file))
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
        try:
            subsets = np.array(list(itertools.combinations(df['Molecule'], 2)))
            indices = {mol: idx for idx, mol in enumerate(df['Molecule'].values)}
            results = np.array([methods[method](x, y, protein_file) for x, y in subsets])
            i, j = np.array([indices[x] for x in subsets[:,0]]), np.array([indices[y] for y in subsets[:,1]])
            matrix = np.zeros((len(df), len(df)))
            matrix[i, j] = results
            matrix[j, i] = results
            matrix_df = pd.DataFrame(matrix, index=df['Pose ID'].values.tolist(), columns=df['Pose ID'].values.tolist())
            clust_df = kmedoids_S_clustering(matrix_df)
            return clust_df
        except Exception as e:
            print(f'Failed to calculate metric and cluster : {e}')

def cluster_numpy_futures(method, w_dir, protein_file):
    print('Loading all poses SDF file...')
    all_poses = PandasTools.LoadSDF(w_dir+'/temp/allposes.sdf', idName='Pose ID', molColName='Molecule', includeFingerprints=False, strictParsing=True)
    print('Finished loading all poses SDF file...')
    id_list = np.unique(np.array(all_poses['ID']))
    create_clustering_folder(w_dir+'/temp/clustering/')
    clustered_dataframes = []
    print(f"*Calculating {method} metrics and clustering*")
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(multiprocessing.cpu_count()/2)) as executor:
        jobs = []
        numMol=0
        for current_id in id_list:
            try:
                job = executor.submit(matrix_calculation_and_clustering_futures, method, all_poses[all_poses['ID']==current_id], protein_file)
                jobs.append(job)
            except Exception as e:
                print("Error in concurrent futures job creation: ", str(e))
            #numMol = numMol+1
        #widgets = [f"Clustering using {method}; ", progressbar.Percentage(), " ", progressbar.ETA(), " ", progressbar.Bar()]
        #pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(jobs))
        #for job in pbar(concurrent.futures.as_completed(jobs)):	
        for job in tqdm(concurrent.futures.as_completed(jobs), total=len(id_list)):
            #try:
                res = job.result()
                clustered_dataframes.append(res)
            #except Exception as e:
                #print("Error in concurrent futures job run: ", str(e))
    clustered_poses = pd.concat(clustered_dataframes)
    clustered_poses['Pose ID'] = clustered_poses['Pose ID'].astype(str).replace('[()\',]','', regex=True)
    clustered_poses = pd.merge(all_poses, clustered_poses, on='Pose ID')
    clustered_poses = clustered_poses[['Pose ID', 'Molecule', 'ID']]
    save_path = w_dir + '/temp/clustering/' + method + '_clustered.sdf'
    PandasTools.WriteSDF(clustered_poses, save_path, molColName='Molecule', idName='Pose ID')
    return