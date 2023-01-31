import pandas as pd
import numpy as np
import math
import os
import random
import functools
from tqdm import tqdm
from spyrmsd import io, rmsd
from espsim import GetEspSim
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import itertools
import rdkit
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from IPython.display import display
from spyrmsd import molecule
from spyrmsd.optional import rdkit as rdkit_loader
import oddt
import oddt.shape
import oddt.fingerprints
import oddt.toolkits.rdk
import multiprocessing
import concurrent.futures

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
    merged_df['Pose ID'] = merged_df['Pose ID'].astype(str).str.replace('[()\',]','', regex=False)
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
    mcs=rdFMCS.FindMCS([mol,jmol])
    pattern = Chem.MolFromSmarts(mcs.smartsString)
    refMatch = mol.GetSubstructMatch(pattern)
    jpattern = jmol.GetSubstructMatch(pattern)
    atomMap = list(zip(jpattern, refMatch))
    rms = AllChem.CalcRMS(mol, jmol, map = [atomMap])
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
                with multiprocessing.Pool(16) as p:
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
def matrix_calculation_and_clustering(method, df_name, protein_file): 
    methods = {'RMSD': simpleRMSD_calc, 'spyRMSD': spyRMSD_calc, 'espsim': espsim_calc, 'USRCAT': USRCAT_calc, 'SPLIF': SPLIF_calc, '3DScore': '3DScore'}
    df_name.index = range(len(df_name['Molecule']))
    table = pd.DataFrame(0.0, index=[df_name['Pose ID']], columns=df_name['Molecule'])
    if method == '3DScore':
        for subset in itertools.combinations(df_name['Molecule'], 2):
            result = methods['spyRMSD'](subset[0], subset[1])
            table.iloc[df_name[df_name['Molecule']==subset[0]].index.values, df_name[df_name['Molecule']==subset[1]].index.values] = 0 if np.isnan(result) else result
            table.iloc[df_name[df_name['Molecule']==subset[1]].index.values, df_name[df_name['Molecule']==subset[0]].index.values] = 0 if np.isnan(result) else result
        table['3DScore'] = table.sum(axis=1)
        table.sort_values(by='3DScore', ascending=True)
        table = table.head(1)
        table.reset_index(inplace=True)
        table = pd.DataFrame(table['Pose ID'])
        table['Pose ID'] = table['Pose ID'].astype(str).str.replace('[()\',]','', regex=False)
        #clustered_dataframes.append(table)
        return table
    else:
        
        for subset in itertools.combinations(df_name['Molecule'], 2):
            #print("subset[0]",str(subset[0])," subset[1]: ", str(subset[1]), "protein_file: ",str(protein_file))
            try:
                result = simpleRMSD_calc(subset[0], subset[1])
            except Exception as e:
                print("Exception in method:", e)
            table.iloc[df_name[df_name['Molecule']==subset[0]].index.values, df_name[df_name['Molecule']==subset[1]].index.values] = 0 if np.isnan(result) else result
            table.iloc[df_name[df_name['Molecule']==subset[1]].index.values, df_name[df_name['Molecule']==subset[0]].index.values] = 0 if np.isnan(result) else result
        clust_df = kmedoids_S_clustering(table)
        clust_df=clust_df['Pose ID']
        return clust_df
        #clustered_dataframes.append(clust_df)
    #full_df = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Pose ID'], how='outer'), clustered_dataframes)
    #full_df['Pose ID'] = full_df['Pose ID'].astype(str).replace('[()\',]','', regex=True)
    #return full_df
def cluster(method, w_dir, protein_file):
    #all_poses = all_poses.groupBy()
    id_list = np.unique(np.array(all_poses['ID']))
    create_clustering_folder(w_dir+'/temp/clustering/')
    clustered_dataframes = []
    print("*Calculating {} metrics and clustering*".format(method))
    methods = {'RMSD': simpleRMSD_calc, 'spyRMSD': spyRMSD_calc, 'espsim': espsim_calc, 'USRCAT': USRCAT_calc, 'SPLIF': SPLIF_calc, '3DScore': '3DScore'}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        jobs = []
        numMol=0
        for current_id in id_list:
            try:
                job = executor.submit(matrix_calculation_and_clustering, methods[method], all_poses[all_poses['ID']==current_id], protein_file)
                jobs.append(job)
            except Exception as e:
                print("Error: ", str(e))
            #numMol = numMol+1
        widgets = ["Generating conformations; ", progressbar.Percentage(), " ", progressbar.ETA(), " ", progressbar.Bar()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(jobs))
        for job in pbar(concurrent.futures.as_completed(jobs)):		
            try:
                res = job.result()
                clustered_dataframes.append(res)
            except Exception as e:
                print("Error: ", str(e))
    full_df = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Pose ID'], how='outer'), clustered_dataframes)
    full_df['Pose ID'] = full_df['Pose ID'].astype(str).replace('[()\',]','', regex=True)
    #return full_df
    #clustered_poses = matrix_calculation_and_clustering(method, all_poses, id_list, protein_file, w_dir)
    clustered_poses = pd.merge(all_poses, full_df, on='Pose ID')
    # keep only the necessary columns
    clustered_poses = clustered_poses[['Pose ID', 'Molecule']]
    save_path = w_dir + '/temp/clustering/' + method + '_clustered.sdf'
    PandasTools.WriteSDF(clustered_poses, save_path, molColName='Molecule', idName='Pose ID')
    return clustered_poses