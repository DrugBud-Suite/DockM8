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

def create_clustering_folder(path):
    if os.path.isdir(path) == True:
        print('Clustering folder already exists')
    else:
        os.mkdir(path)
        print('Clustering folder was created')

def kmedoids_S_clustering(input_dataframe):
    df = input_dataframe.copy()
    # Get Pose ID and Molecule names for renaming
    molecule_list = input_dataframe.columns.values.tolist()
    id_list = input_dataframe.index.tolist()
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
    df['KMedoids Cluster']=clusters
    df['Molecule'] = molecule_list
    df['Pose ID'] = id_list
    # Determine centers
    centroids = kmedoids.cluster_centers_
    cluster_centers = pd.DataFrame(centroids,columns = molecule_list)
    #rearranging data
    merged_df = pd.merge(df, cluster_centers, on=molecule_list)
    merged_df = merged_df[['Molecule', 'Pose ID']]
    merged_df['Pose ID'] = merged_df['Pose ID'].astype(str).replace('[()\',]','', regex=False)
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
def symmRMSD_calc(mol, jmol):
    rms = Chem.rdMolAlign.CalcRMS(mol, jmol)
    return round(rms, 3)

def cluster_multiprocessing(method, w_dir, protein_file):
    def matrix_calculation_and_clustering_multiprocessing(method, df, id_list, protein_file, w_dir): 
        matrix = dict()
        clustered_dataframes = []
        print("*Calculating {} metrics and clustering*".format(method))
        methods = {'RMSD': simpleRMSD_calc, 'spyRMSD': spyRMSD_calc, 'espsim': espsim_calc, 'USRCAT': USRCAT_calc, 'SPLIF': SPLIF_calc}
        for id in tqdm(id_list):
            if method == '3DScore':
                df_name = df[df['ID']==id]
                df_name.index = range(len(df_name['Molecule']))
                table = pd.DataFrame(0.0, index=[df_name['Pose ID']], columns=df_name['Molecule'])
                with multiprocessing.Pool() as p:
                        try:
                            results = p.map(methods['spyRMSD'], itertools.combinations(df_name['Molecule'], 2))
                        except KeyError:
                            print('Incorrect clustering method selected')
                            return
                        results_list = list(zip(itertools.combinations(df_name['Molecule'], 2), results))
                        for subset, result in results_list:
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
                    with multiprocessing.Pool() as p:
                        try:
                            results = p.map(methods[method], itertools.combinations(df_name['Molecule'], 2))
                        except KeyError:
                            print('Incorrect clustering method selected')
                            return
                        results_list = list(zip(itertools.combinations(df_name['Molecule'], 2), results))
                        for subset, result in results_list:
                            table.iloc[df_name[df_name['Molecule']==subset[0]].index.values, df_name[df_name['Molecule']==subset[1]].index.values] = 0 if np.isnan(result) else result
                            table.iloc[df_name[df_name['Molecule']==subset[1]].index.values, df_name[df_name['Molecule']==subset[0]].index.values] = 0 if np.isnan(result) else result
                    matrix[id+'_'+method] = table
                    clust_df = kmedoids_S_clustering(table)
                    clust_df=clust_df['Pose ID']
                    clustered_dataframes.append(clust_df)
                except:
                    print(f'Failed to calculate metrics and cluster ID: {id}')
        full_df = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Pose ID'], how='outer'), clustered_dataframes)
        return full_df
    print('Loading all poses SDF file...')
    all_poses = PandasTools.LoadSDF(w_dir+'/temp/allposes.sdf', idName='Pose ID', molColName='Molecule', includeFingerprints=False, embedProps=True, removeHs=True, strictParsing=True)
    print('Finished loading all poses SDF file...')
    id_list = np.unique(np.array(all_poses['ID']))
    create_clustering_folder(w_dir+'/temp/clustering/')
    clustered_poses = matrix_calculation_and_clustering_multiprocessing(method, all_poses, id_list, protein_file, w_dir)
    clustered_poses = pd.merge(all_poses, clustered_poses, on='Pose ID')
    # keep only the necessary columns
    clustered_poses = clustered_poses[['Pose ID', 'Molecule']]
    save_path = w_dir + '/temp/clustering/' + method + '_clustered.sdf'
    PandasTools.WriteSDF(clustered_poses, save_path, molColName='Molecule', idName='Pose ID')
    return

def cluster(method, w_dir, protein_file):
    def matrix_calculation_and_clustering(method, df, id_list, protein_file, w_dir): 
        matrix = dict()
        clustered_dataframes = []
        print("*Calculating {} metrics and clustering*".format(method))
        methods = {'RMSD': simpleRMSD_calc, 'spyRMSD': spyRMSD_calc, 'espsim': espsim_calc, 'USRCAT': USRCAT_calc, 'SPLIF': SPLIF_calc, '3DScore': '3DScore', 'bestpose': 'bestpose'}
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
                    matrix[id+'_'+method] = table
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
        matrix = dict()
        clustered_dataframes = []
        print("*Calculating {} metrics and clustering*".format(method))
        methods = {'RMSD': simpleRMSD_calc, 'spyRMSD': spyRMSD_calc, 'espsim': espsim_calc, 'USRCAT': USRCAT_calc, 'SPLIF': SPLIF_calc, '3DScore': '3DScore', 'bestpose': 'bestpose'}
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
                for subset in itertools.combinations(df_name['Molecule'], 2):
                    try:
                        result = methods['spyRMSD'](subset[0], subset[1])
                    except:
                        result = methods['RMSD'](subset[0], subset[1])
                    table.iloc[df_name[df_name['Molecule']==subset[0]].index.values, df_name[df_name['Molecule']==subset[1]].index.values] = 0 if np.isnan(result) else result
                    table.iloc[df_name[df_name['Molecule']==subset[1]].index.values, df_name[df_name['Molecule']==subset[0]].index.values] = 0 if np.isnan(result) else result
                
                df_filtered = df[df['ID']==id]
                matrix = np.zeros((len(df_filtered), len(df_filtered)))
                subsets = np.array(list(itertools.combinations(df_filtered['Molecule'], 2)))
                subset1 = subsets[:,0]
                subset2 = subsets[:,1]
                indices = {mol: idx for idx, mol in enumerate(df_filtered['Molecule'].values)}
                for x, y in subsets:
                    try:
                        results = np.array([methods['spyRMSD'](x, y, protein_file)])
                    except:
                        results = np.array([methods['RMSD'](x, y, protein_file)])
                i, j = np.array([indices[x] for x in subset1]), np.array([indices[y] for y in subset2])
                matrix[i, j] = results
                matrix[j, i] = results
                matrix_df = pd.DataFrame(matrix)
                matrix_df['3DScore'] = matrix_df.sum(axis=1)
                matrix_df.sort_values(by='3DScore', ascending=True)
                matrix_df = matrix_df.head(1)
                matrix_df.reset_index(inplace=True)
                matrix_df = pd.DataFrame(matrix_df['Pose ID'])
                matrix_df['Pose ID'] = matrix_df['Pose ID'].astype(str).str.replace('[()\',]','', regex=False)
                clustered_dataframes.append(matrix_df)
            else:
                #try:
                    df_filtered = df[df['ID']==id]
                    matrix = np.zeros((len(df_filtered), len(df_filtered)))
                    subsets = np.array(list(itertools.combinations(df_filtered['Molecule'], 2)))
                    subset1 = subsets[:,0]
                    subset2 = subsets[:,1]
                    indices = {mol: idx for idx, mol in enumerate(df_filtered['Molecule'].values)}
                    results = np.array([methods[method](x, y, protein_file) for x, y in subsets])
                    i, j = np.array([indices[x] for x in subset1]), np.array([indices[y] for y in subset2])
                    matrix[i, j] = results
                    matrix[j, i] = results
                    matrix_df = pd.DataFrame(matrix)
                    matrix_df.columns = [df['Pose ID']]
                    matrix_df.index = [df['Pose ID']]
                    clust_df = kmedoids_S_clustering(matrix_df)
                    clust_df = clust_df['Pose ID']
                    clustered_dataframes.append(clust_df)
                #except:
                    #print(f'Failed to calculate metrics and cluster ID: {id}')
        full_df = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Pose ID'], how='outer'), clustered_dataframes)
        full_df['Pose ID'] = full_df['Pose ID'].astype(str).replace('[()\',]','', regex=True)
        return full_df
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

def cluster_dask(method, w_dir, protein_file):
    create_clustering_folder(w_dir+'/temp/clustering/')
    def matrix_calculation_and_clustering(method, df, id_list, protein_file, w_dir): 
        #print("*Calculating {} metrics and clustering*".format(method))
        id_list = np.unique(np.array(df['Pose ID']))
        methods = {'RMSD': simpleRMSD_calc, 'spyRMSD': spyRMSD_calc, 'espsim': espsim_calc, 'USRCAT': USRCAT_calc, 'SPLIF': SPLIF_calc, '3DScore': '3DScore', 'bestpose': 'bestpose'}
        if method == 'bestpose':
            df[['CHEMPLP', 'SMINA_Affinity', 'CNNaffinity']] = df[['CHEMPLP', 'SMINA_Affinity', 'CNNaffinity']].apply(pd.to_numeric, errors='coerce')
            best_row_CHEMPLP = df.loc[df.groupby(['Pose ID'])['CHEMPLP'].idxmin()]
            best_row_SMINA = df.loc[df.groupby(['Pose ID'])['SMINA_Affinity'].idxmin()]
            best_row_GNINA = df.loc[df.groupby(['Pose ID'])['CNNaffinity'].idxmax()]
            table = pd.concat([best_row_GNINA, best_row_SMINA, best_row_CHEMPLP])
            table.reset_index(inplace=True)
            table = pd.DataFrame(table['Pose ID'])
            table['Pose ID'] = table['Pose ID'].astype(str).str.replace('[()\',]','', regex=False)
        return table

    print('Loading all poses SDF file...')
    all_poses = PandasTools.LoadSDF(w_dir+'/temp/allposes.sdf', idName='Pose ID', molColName='Molecule', includeFingerprints=False, strictParsing=True)
    print('Finished loading all poses SDF file...')
    id_list = np.unique(np.array(all_poses['ID']))
    dask_df = dd.from_pandas(all_poses, npartitions=multiprocessing.cpu_count())
    grouped_df = dask_df.groupby('ID')
    results = grouped_df.map(matrix_calculation_and_clustering, method, all_poses, id_list, protein_file, w_dir)
    print(results)
    #clustered_poses = matrix_calculation_and_clustering(method, all_poses, id_list, protein_file, w_dir)
    clustered_poses = pd.merge(all_poses, clustered_poses, on='Pose ID')
    # keep only the necessary columns
    clustered_poses = clustered_poses[['Pose ID', 'Molecule', 'ID']]
    save_path = w_dir + '/temp/clustering/' + method + '_clustered.sdf'
    PandasTools.WriteSDF(clustered_poses, save_path, molColName='Molecule', idName='Pose ID')
    return