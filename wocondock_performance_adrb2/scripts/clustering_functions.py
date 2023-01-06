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

def create_clustering_folder(path):
    if os.path.isdir(path) == True:
        print('Clustering folder already exists')
    else:
        os.mkdir(path)
        print('Clustering folder was created')

def kmedoids_S_clustering(input_dataframe):
    dataframe = input_dataframe.copy()
    # Get Pose ID and Molecule names for renaming
    column_list = input_dataframe.columns.values.tolist()
    index_list = input_dataframe.index.tolist()
    #preprocessing our data *scaling data* 
    scaler = StandardScaler()
    dataframe[column_list] = scaler.fit_transform(dataframe)
    # Calculate maximum number of clusters, limited to 5 centroids
    max_clusters = 5
    range_n_clusters = range(2, max_clusters)
    # Define function for silhouette calculation
    silhouettes_scores = []
    for n in range_n_clusters:
        #calculating silhouette average score for every cluster and plotting them at the end
        #choosing pam method as it's more accurate
        # initialization of medoids is a greedy approach, as it's more effecient as well 
        kmedoids = KMedoids(n_clusters=n , method='pam',init='build' ,max_iter=150)
        cluster_labels = kmedoids.fit_predict(dataframe)
        silhouette_average_score = silhouette_score(dataframe, cluster_labels)
        silhouettes_scores.append(silhouette_average_score)
    silhouette_df = pd.DataFrame(columns=['number of clusters', 'silhouette_score'])
    silhouette_df['number of clusters'] = range_n_clusters
    silhouette_df['silhouette_score'] = silhouettes_scores
    optimum_no_clusters = int(silhouette_df.loc[silhouette_df['silhouette_score'].idxmax(), ['number of clusters']])
    # # Apply optimised k-medoids clustering
    kmedoids = KMedoids(n_clusters=optimum_no_clusters, method='pam',init='build', max_iter=150)
    clusters = kmedoids.fit_predict(dataframe)
    dataframe['KMedoids Cluster']=clusters
    dataframe['Molecule'] = column_list
    dataframe['Pose ID'] = index_list
    # Determine centers
    centroids = kmedoids.cluster_centers_
    cluster_centers = pd.DataFrame(centroids,columns = column_list)
    cluster_centers['Pose ID'] = "NaN"
    cluster_centers['Molecule'] = "NaN"
    #rearranging data
    merged_df = pd.merge(dataframe, cluster_centers, left_on=column_list, right_on=column_list)
    merged_df.rename(columns = {'Molecule_x':'Molecule', 'Pose ID_x':'Pose ID'}, inplace = True)
    merged_df = merged_df.loc[:, ['Molecule', 'Pose ID']]
    merged_df['Pose ID'] = merged_df['Pose ID'].astype('string')
    merged_df['Pose ID'] = merged_df['Pose ID'].str.replace(')','', regex=False)
    merged_df['Pose ID'] = merged_df['Pose ID'].str.replace('(','', regex=False)
    merged_df['Pose ID'] = merged_df['Pose ID'].str.replace(',','', regex=False)
    merged_df['Pose ID'] = merged_df['Pose ID'].str.replace("'",'', regex=False)
    return merged_df

def simpleRMSD(mol, jmol):
# MCS identification between reference pose and target pose
    r=rdFMCS.FindMCS([mol,jmol])
# Atom map for reference and target              
    a=mol.GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
    b=jmol.GetSubstructMatch(Chem.MolFromSmarts(r.smartsString))
# Atom map generation     
    amap=list(zip(a,b))
# distance calculation per atom pair
    distances=[]
    for atomA, atomB in amap:
        pos_A=mol.GetConformer().GetAtomPosition (atomA)
        pos_B=jmol.GetConformer().GetAtomPosition (atomB)
        coord_A=np.array((pos_A.x,pos_A.y,pos_A.z))
        coord_B=np.array ((pos_B.x,pos_B.y,pos_B.z))
        dist_numpy = np.linalg.norm(coord_A-coord_B)        
        distances.append(dist_numpy)      
# This is the RMSD formula from wikipedia
    rmsd=math.sqrt(1/len(distances)*sum([i*i for i in distances])) 
    return round(rmsd, 3)

def spyRMSD(rmol, rjmol):
    #mol=rdkit_loader.load(rmol)
    #jmol=rdkit_loader.load(rjmol)
    spyrmsd_mol = molecule.Molecule.from_rdkit(rmol)
    spyrmsd_jmol = molecule.Molecule.from_rdkit(rjmol)
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

def SPLIF_calc(mol, jmol, protein_file):
    pocket_file = protein_file.replace('.pdb', '_pocket.pdb')
    protein=next(oddt.toolkit.readfile('pdb', pocket_file))
    protein.protein = True
    splif_mol = oddt.toolkits.rdk.Molecule(mol)
    splif_jmol = oddt.toolkits.rdk.Molecule(jmol)
    mol_fp=oddt.fingerprints.SimpleInteractionFingerprint(splif_mol, protein)
    jmol_fp=oddt.fingerprints.SimpleInteractionFingerprint(splif_jmol, protein)
    SPLIF_sim = oddt.fingerprints.tanimoto(mol_fp, jmol_fp)
    return round(SPLIF_sim, 3)

def usr_cat_calc(mol, jmol):
    shape_mol = oddt.toolkits.rdk.Molecule(mol)
    shape_jmol = oddt.toolkits.rdk.Molecule(jmol)
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

def matrix_calculation(method, df, id_list, protein_file): 
    matrix = dict()
    print("*Calculating {} metrics*".format(method))
    print(id_list)
    for id in tqdm(id_list): 
        df_name = 'df_'+id
        df_name = df[df['ID']==id]
        df_name.index = range(len(df_name['Molecule']))
        #intialize dataframe for Espsim
        table1 = pd.DataFrame(0.0, index=[df_name['Pose ID']], columns=df_name['Molecule'])
        table2 = pd.DataFrame(0.0, index=[df_name['Pose ID']], columns=df_name['Molecule'])
        table3 = pd.DataFrame(0.0, index=[df_name['Pose ID']], columns=df_name['Molecule'])
        table4 = pd.DataFrame(0.0, index=[df_name['Pose ID']], columns=df_name['Molecule'])
        table5 = pd.DataFrame(0.0, index=[df_name['Pose ID']], columns=df_name['Molecule'])
        table6 = pd.DataFrame(0.0, index=[df_name['Pose ID']], columns=df_name['Molecule'])
        #finding unique combinations only and calculate RMSD & Espsim between them
        for subset in tqdm(itertools.combinations(df_name['Molecule'], 2), total=df_name.shape[0]):
            if 'RMSD' in method:
                #adding RMSD score in the right position
                rmsd = simpleRMSD(subset[0], subset[1])
                table1.iloc[df_name[df_name['Molecule']==subset[0]].index.values, df_name[df_name['Molecule']==subset[1]].index.values] = rmsd
                table1.iloc[df_name[df_name['Molecule']==subset[1]].index.values, df_name[df_name['Molecule']==subset[0]].index.values] = rmsd
            elif 'spyRMSD' in method:
                #adding spyRMSD score in the right position
                spyrmsd = spyRMSD(subset[0], subset[1])
                table3.iloc[df_name[df_name['Molecule']==subset[0]].index.values, df_name[df_name['Molecule']==subset[1]].index.values] = spyrmsd
                table3.iloc[df_name[df_name['Molecule']==subset[1]].index.values, df_name[df_name['Molecule']==subset[0]].index.values] = spyrmsd
            elif 'espsim' in method:
                #adding espsim score in the right position
                espsim = float(GetEspSim(subset[0], subset[1]))
                table2.iloc[df_name[df_name['Molecule']==subset[0]].index.values, df_name[df_name['Molecule']==subset[1]].index.values] = round(espsim, 3)
                table2.iloc[df_name[df_name['Molecule']==subset[1]].index.values, df_name[df_name['Molecule']==subset[0]].index.values] = round(espsim, 3)
            elif 'SPLIF' in method:
                #adding SPLIF score in the right position
                splif_sim = SPLIF_calc(subset[0], subset[1], protein_file)
                table4.iloc[df_name[df_name['Molecule']==subset[0]].index.values, df_name[df_name['Molecule']==subset[1]].index.values] = splif_sim
                table4.iloc[df_name[df_name['Molecule']==subset[1]].index.values, df_name[df_name['Molecule']==subset[0]].index.values] = splif_sim
            elif 'USRCAT' in method:
                #adding USRCAT score in the right position
                usr_sim = usr_cat_calc(subset[0], subset[1])
                table5.iloc[df_name[df_name['Molecule']==subset[0]].index.values, df_name[df_name['Molecule']==subset[1]].index.values] = usr_sim
                table5.iloc[df_name[df_name['Molecule']==subset[1]].index.values, df_name[df_name['Molecule']==subset[0]].index.values] = usr_sim
            elif 'symmRMSD' in method:
                #adding symmRMSD score in the right position
                rms = symmRMSD_calc(subset[0], subset[1])
                table5.iloc[df_name[df_name['Molecule']==subset[0]].index.values, df_name[df_name['Molecule']==subset[1]].index.values] = rms
                table5.iloc[df_name[df_name['Molecule']==subset[1]].index.values, df_name[df_name['Molecule']==subset[0]].index.values] = rms
            else:
                print('Incorrect clustering method selected')    
        matrix[id+'_SimpleRMSD'] = table1
        matrix[id+'_SpyRMSD'] = table3
        matrix[id+'_espsim'] = table2
        matrix[id+'_SPLIF'] = table4
        matrix[id+'_USRCAT'] = table5
        matrix[id+'_symmRMSD'] = table6
    print("Ready for Clustering!!")
    return matrix

def merge_centroids(method, w_dir, id_list, matrix):
    print("Clustering using K-Medoids...")
    RMSD_kS_full_path = w_dir+'/temp/clustering/RMSD_kS_full.sdf'
    espsim_d_kS_full_path = w_dir+'/temp/clustering/espsim_d_kS_full.sdf'
    spyRMSD_kS_full_path = w_dir+'/temp/clustering/spyRMSD_kS_full.sdf'
    splif_kS_full_path = w_dir+'/temp/clustering/SPLIF_kS_full.sdf'
    usr_kS_full_path = w_dir+'/temp/clustering/usr_kS_full.sdf'
    symmRMSD_kS_full_path = w_dir+'/temp/clustering/usr_kS_full.sdf'
    RMSD_kS_full = pd.DataFrame()
    spyRMSD_kS_full = pd.DataFrame()
    espsim_d_kS_full = pd.DataFrame()
    splif_kS_full = pd.DataFrame()
    usr_kS_full = pd.DataFrame()
    symmRMSD_kS_full = pd.DataFrame()
    rmsd_dataframes = []
    spyrmsd_dataframes = []
    espsim_dataframes = []
    splif_dataframes = []
    usr_dataframes = []
    symmRMSD_dataframes = []

    #collecting clustered dataframes for each function in a list
    for id in tqdm(id_list):
        if 'RMSD' in method:
            rmsd_df = kmedoids_S_clustering(matrix[id+'_SimpleRMSD'])
            rmsd_dataframes.append(rmsd_df)
        elif 'espsim' in method:
            espsim_df = kmedoids_S_clustering(matrix[id+'_espsim'])
            espsim_dataframes.append(espsim_df)
        elif 'spyRMSD' in method:
            spyrmsd_df = kmedoids_S_clustering(matrix[id+'_SpyRMSD'])
            spyrmsd_dataframes.append(spyrmsd_df)
        elif 'SPLIF' in method:
            splif_df = kmedoids_S_clustering(matrix[id+'_SPLIF'])
            splif_dataframes.append(splif_df)
        elif 'USRCAT' in method:
            usr_df = kmedoids_S_clustering(matrix[id+'_USRCAT'])
            usr_dataframes.append(usr_df)
        elif 'symmRMSD' in method:
            symmRMSD_df = kmedoids_S_clustering(matrix[id+'_symmRMSD'])
            symmRMSD_dataframes.append(symmRMSD_df)
        else:
            print('Incorrect clustering method selected') 
    #merging all centroids found for all compounds according to the function
    print('Merging cluster centers...')
    if 'RMSD' in method:
        RMSD_kS_full = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Pose ID','Molecule'], how='outer'), rmsd_dataframes)
    if 'espsim' in method:
        espsim_d_kS_full = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Pose ID', 'Molecule'], how='outer'), espsim_dataframes)
    if 'spyRMSD' in method:
        spyRMSD_kS_full = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Pose ID','Molecule'], how='outer'), spyrmsd_dataframes)
    if 'SPLIF' in method:
        splif_kS_full = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Pose ID','Molecule'], how='outer'), splif_dataframes)
    if 'USRCAT' in method:
        usr_kS_full = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Pose ID','Molecule'], how='outer'), usr_dataframes)
    if 'symmRMSD' in method:
        symmRMSD_kS_full = functools.reduce(lambda  left,right: pd.merge(left,right,on=['Pose ID','Molecule'], how='outer'), symmRMSD_dataframes)
        #writing in a sdf file in clustering folder
    if 'RMSD' in method:
        PandasTools.WriteSDF(RMSD_kS_full, RMSD_kS_full_path, molColName='Molecule', idName='Pose ID')
    if 'espsim' in method:    
        PandasTools.WriteSDF(espsim_d_kS_full, espsim_d_kS_full_path, molColName='Molecule', idName='Pose ID')
    if 'spyRMSD' in method:
        PandasTools.WriteSDF(spyRMSD_kS_full, spyRMSD_kS_full_path, molColName='Molecule', idName='Pose ID')
    if 'SPLIF' in method:
        PandasTools.WriteSDF(splif_kS_full, splif_kS_full_path, molColName='Molecule', idName='Pose ID')
    if 'USRCAT' in method:
        PandasTools.WriteSDF(usr_kS_full, usr_kS_full_path, molColName='Molecule', idName='Pose ID')
    if 'symmRMSD' in method:
        PandasTools.WriteSDF(symmRMSD_kS_full, symmRMSD_kS_full_path, molColName='Molecule', idName='Pose ID')
    print('Clustering finished!')
    
def cluster(method, w_dir, protein_file, all_poses):
    id_list = np.unique(np.array(all_poses['ID']))
    create_clustering_folder(w_dir+'/temp/clustering/')
    matrix = matrix_calculation(method, all_poses, id_list, protein_file)
    merge_centroids(method, w_dir, id_list, matrix)
    return
    