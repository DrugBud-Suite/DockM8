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
parser.add_argument('--pocket', type = str, choices = ['reference', 'dogsitescorer'], help ='Method to use for pocket determination')
parser.add_argument('--reffile', type=str, help ='Path to reference ligand file')
parser.add_argument('--dockinglibrary', required=True, type=str, help ='Path to docking library file')
parser.add_argument('--idcolumn', required=True, type=str, help ='Unique identifier column')
parser.add_argument('--protonation', required=True, type = str, choices = ['pkasolver', 'GypsumDL', 'None'], help ='Method to use for compound protonation')
parser.add_argument('--docking', type = str, nargs='+', choices = ['GNINA', 'SMINA', 'PLANTS'], help ='Method(s) to use for docking')
parser.add_argument('--clustering', type = str, nargs='+', choices = ['RMSD', 'spyRMSD', 'espsim', 'USRCAT', '3DScore', 'bestscore'], help ='Method(s) to use for pose clustering')
parser.add_argument('--nposes', default=10, type=int, help ='Number of poses')
parser.add_argument('--exhaustiveness', default=8, type = int, help ='Precision of SMINA/GNINA')

args = parser.parse_args()

if args.pocket == 'reference' and not args.reffile:
    parser.error("--reffile is required when --pocket is set to 'reference'")

def run_command(**kwargs):
    w_dir = os.path.dirname(kwargs.get('proteinfile'))
    print('The working directory has been set to:', w_dir)
    create_temp_folder(w_dir+'/temp')
    if kwargs.get('pocket') == 'reference':
        pocket_definition = GetPocket(kwargs.get('reffile'), kwargs.get('proteinfile'), 8)
    elif kwargs.get('pocket') == 'dogsitescorer':
        pocket_definition = binding_site_coordinates_dogsitescorer(kwargs.get('proteinfile'), w_dir, method='volume')
        
    if Path(w_dir+'/temp/final_library.sdf').isfile() == False:
        cleaned_df = prepare_library(kwargs.get('dockinglibrary'), kwargs.get('idcolumn'), kwargs.get('software'), kwargs.get('protonation'))

    if Path(w_dir+'/temp/all_poses.sdf').isfile() == False:
        all_poses = docking(kwargs.get('proteinfile'), kwargs.get('reffile'), kwargs.get('software'), kwargs.get('docking'), kwargs.get('exhaustiveness'), kwargs.get('nposes'))
    
    clustering_methods =kwargs.get('clustering').split(' ')
    for method in clustering_methods:
        if Path(w_dir+f'/temp/clustering/{method}_clustered.sdf').isfile() == False:
            cluster(method, w_dir, kwargs.get('proteinfile'))
    
    for method in clustering_methods:
        if Path(w_dir+f'/temp/rescoring_{method}_clustered').ispath() == False:
            rescore_all(w_dir, kwargs.get('proteinfile'), kwargs.get('reffile'), kwargs.get('software'), w_dir+f'/temp/clustering/{method}_clustered.sdf')

    apply_ranking_methods_simplified(w_dir)
    
    calculate_EFs(w_dir, kwargs.get('dockinglibrary'))
    
run_command(**vars(args))