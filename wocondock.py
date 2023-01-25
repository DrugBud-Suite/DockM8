#Import required libraries and scripts
from scripts.library_preparation import *
from scripts.utilities import *
from scripts.docking_functions import *
from scripts.clustering_functions import *
from scripts.rescoring_functions import *
from scripts.ranking_functions import *
from scripts.get_pocket import *
import numpy as np
import os

def run_command(software, protein_file, ref_file, docking_library, id_column, n_poses=10, exhaustiveness=8):
    w_dir = os.path.dirname(protein_file)
    print('The working directory has been set to:', w_dir)
    create_temp_folder(w_dir+'/temp')
    
    pocket = GetPocket(ref_file, protein_file, 8)
    
    cleaned_df = prepare_library_GypsumDL(docking_library, id_column, software)
    
    all_poses = docking(protein_file, ref_file, id_column, software, exhaustiveness, n_poses)
    
    cluster(['RMSD, spyRMSD, espsim, USRCAT'], w_dir, id_column, protein_file)
    
    RMSD_rescored = rescore_all(w_dir, protein_file, ref_file, software, w_dir+'/temp/clustering/RMSD_kS_full.sdf')
    espsim_rescored = rescore_all(w_dir, protein_file, ref_file, software, w_dir+'/temp/clustering/espsim_d_kS_full.sdf')
    spyRMSD_rescored = rescore_all(w_dir, protein_file, ref_file, software, w_dir+'/temp/clustering/spyRMSD_kS_full.sdf')
    USRCAT_rescored = rescore_all(w_dir, protein_file, ref_file, software, w_dir+'/temp/clustering/usr_kS_full.sdf')
    
    RMSD_rescored_ranked = rank(RMSD_rescored)
    espsim_rescored_ranked = rank(espsim_rescored)
    spyRMSD_rescored_ranked = rank(spyRMSD_rescored)
    USRCAT_rescored_ranked = rank(USRCAT_rescored)
    
    method1, method2, method3 = apply_all_rank_methods(RMSD_rescored_ranked)
    method6, method7 = apply_all_score_methods(RMSD_rescored)
    methods_dfs = [method1, method2, method3, method6, method7]
    combined_methods_dfs = functools.reduce(lambda  left,right: pd.merge(left,right,on=id_column,how='outer'), methods_dfs)
    combined_methods_dfs = combined_methods_dfs.drop(['Pose ID_x', 'Pose ID_y'], axis=1)
    combined_methods_dfs