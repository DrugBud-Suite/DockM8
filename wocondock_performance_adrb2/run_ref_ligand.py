#Import required libraries and scripts
from scripts.library_preparation import *
from scripts.utilities import *
from scripts.docking_functions import *
from scripts.clustering_functions import *
from scripts.rescoring_functions import *
from scripts.ranking_functions import *
from scripts.get_pocket import *
from scripts.performance_calculation import *
from IPython.display import display
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Parse required arguments')
parser.add_argument('--software', help='Path to software folder')
parser.add_argument('--proteinfile', help='Path to protein file')
parser.add_argument('--reffile', help='Path to reference ligand file')
parser.add_argument('--dockinglibrary', help='Path to docking library file')
parser.add_argument('--idcolumn', help='Unique identifier column')
parser.add_argument('--nposes', type=int, help='Number of poses')
parser.add_argument('--exhaustiveness', type = int, help='Precision of SMINA/GNINA')
args = parser.parse_args()

software_arg = args.software
protein_arg = args.proteinfile
ref_arg = args.reffile
library_arg = args.dockinglibrary
idcolumn_arg = args.idcolumn
nposes_arg = args.nposes
exhaustiveness_arg = args.exhaustiveness

def run_command(software, protein_file, ref_file, docking_library, id_column, n_poses=10, exhaustiveness=8):
    w_dir = os.path.dirname(protein_file)
    print('The working directory has been set to:', w_dir)
    create_temp_folder(w_dir+'/temp')
    
    pocket = GetPocket(ref_file, protein_file, 8)
    
    cleaned_df = prepare_library_GypsumDL(docking_library, id_column, software)
    
    all_poses = docking(protein_file, ref_file, software, exhaustiveness, n_poses)
    
    display(all_poses)
    
    cluster(['RMSD'], w_dir, protein_file, all_poses)
    cluster(['espsim'], w_dir, protein_file, all_poses)
    cluster(['spyRMSD'], w_dir, protein_file, all_poses)
    cluster(['USRCAT'], w_dir, protein_file, all_poses)
    
    RMSD_rescored = rescore_all(w_dir, protein_file, ref_file, software, w_dir+'/temp/clustering/RMSD_kS_full.sdf')
    espsim_rescored = rescore_all(w_dir, protein_file, ref_file, software, w_dir+'/temp/clustering/espsim_d_kS_full.sdf')
    spyRMSD_rescored = rescore_all(w_dir, protein_file, ref_file, software, w_dir+'/temp/clustering/spyRMSD_kS_full.sdf')
    USRCAT_rescored = rescore_all(w_dir, protein_file, ref_file, software, w_dir+'/temp/clustering/usr_kS_full.sdf')
    
    apply_ranking_methods(w_dir)
    
    calculate_EFs(w_dir, docking_library)
    
run_command(software_arg, protein_arg, ref_arg, library_arg, idcolumn_arg, nposes_arg, exhaustiveness_arg)