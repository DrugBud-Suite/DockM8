#Import required libraries and scripts
from scripts.library_preparation import *
from scripts.utilities import *
from scripts.docking_functions import *
from scripts.clustering_functions import *
from scripts.rescoring_functions import *
from scripts.ranking_functions import *
from scripts.get_pocket import *
from scripts.dogsitescorer import *
from scripts.performance_calculation import *
from IPython.display import display
from pathlib import Path
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Parse required arguments')
parser.add_argument('--software', required=True, type=str, help ='Path to software folder')
parser.add_argument('--proteinfile', required=True, type=str, help ='Path to protein file')
parser.add_argument('--pocket', required=True, type = str, choices = ['reference', 'dogsitescorer'], help ='Method to use for pocket determination')
parser.add_argument('--reffile', type=str, help ='Path to reference ligand file')
parser.add_argument('--dockinglibrary', required=True, type=str, help ='Path to docking library file')
parser.add_argument('--idcolumn', required=True, type=str, help ='Unique identifier column')
parser.add_argument('--protonation', required=True, type = str, choices = ['pkasolver', 'GypsumDL', 'None'], help ='Method to use for compound protonation')
parser.add_argument('--docking', required=True, type = str, nargs='+', choices = ['GNINA', 'SMINA', 'PLANTS'], help ='Method(s) to use for docking')
parser.add_argument('--clustering', required=True, type = str, nargs='+', choices = ['RMSD', 'spyRMSD', 'espsim', 'USRCAT', '3DScore', 'bestpose'], help ='Method(s) to use for pose clustering')
parser.add_argument('--nposes', default=10, type=int, help ='Number of poses')
parser.add_argument('--exhaustiveness', default=8, type = int, help ='Precision of SMINA/GNINA')
parser.add_argument('--parallel', default=1, type=int, choices = [0,1], help ='Run the workflow in parallel')

args = parser.parse_args()

if args.pocket == 'reference' and not args.reffile:
    parser.error("--reffile is required when --pocket is set to 'reference'")

def run_command(**kwargs):
    w_dir = os.path.dirname(kwargs.get('proteinfile'))
    print('The working directory has been set to:', w_dir)
    create_temp_folder(w_dir+'/temp')
    
    if os.path.isfile(kwargs.get('proteinfile').replace('.pdb', '_pocket.pdb')) == False:
        if kwargs.get('pocket') == 'reference':
            pocket_definition = GetPocket(kwargs.get('reffile'), kwargs.get('proteinfile'), 8)
        elif kwargs.get('pocket') == 'dogsitescorer':
            pocket_definition = binding_site_coordinates_dogsitescorer(kwargs.get('proteinfile'), w_dir, method='volume')
            
    if os.path.isfile(w_dir+'/temp/final_library.sdf') == False:
        prepare_library(kwargs.get('dockinglibrary'), kwargs.get('idcolumn'), kwargs.get('software'), kwargs.get('protonation'))

    if kwargs.get('parallel') == 0:
        docking_func = docking
        fetch_poses_func = fetch_poses
        cluster_func = cluster_numpy
    else:
        docking_func = docking_splitted
        fetch_poses_func = fetch_poses_splitted
        cluster_func = cluster_numpy_futures

    docking_programs = {'GNINA': w_dir+'/temp/gnina/', 'SMINA': w_dir+'/temp/smina/', 'PLANTS': w_dir+'/temp/plants/'}
    for program, file_path in docking_programs.items():
        if os.path.isdir(file_path) == False and program in kwargs.get('docking'):
            docking_func(w_dir, kwargs.get('proteinfile'), kwargs.get('reffile'), kwargs.get('software'), [program], kwargs.get('exhaustiveness'), kwargs.get('nposes'))

    if os.path.isfile(w_dir+'/temp/allposes.sdf') == False:
        fetch_poses_func(w_dir, kwargs.get('nposes'), w_dir+'/temp/split_final_library')

    for method in kwargs.get('clustering'):
        if os.path.isfile(w_dir+f'/temp/clustering/{method}_clustered.sdf') == False:
            cluster_func(method, w_dir, kwargs.get('proteinfile'))
    
    for method in kwargs.get('clustering'):
        if os.path.isdir(w_dir+f'/temp/rescoring_{method}_clustered') == False:
            rescore_all(w_dir, kwargs.get('proteinfile'), kwargs.get('reffile'), kwargs.get('software'), w_dir+f'/temp/clustering/{method}_clustered.sdf')

    apply_ranking_methods_simplified(w_dir)
    
    calculate_EFs(w_dir, kwargs.get('dockinglibrary'))
    
run_command(**vars(args))