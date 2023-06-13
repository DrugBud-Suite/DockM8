#Import required libraries and scripts
from scripts.library_preparation import *
from scripts.utilities import *
from scripts.docking_functions import *
from scripts.clustering_functions import *
from scripts.rescoring_functions import *
from scripts.consensus_methods import *
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
parser.add_argument('--metric', required=True, type = str, nargs='+', choices = ['RMSD', 'spyRMSD', 'espsim', 'USRCAT', '3DScore', 'bestpose', 'bestpose_GNINA', 'bestpose_SMINA', 'bestpose_PLANTS'], help ='Method(s) to use for pose clustering')
parser.add_argument('--nposes', default=10, type=int, help ='Number of poses')
parser.add_argument('--exhaustiveness', default=8, type = int, help ='Precision of SMINA/GNINA')
parser.add_argument('--ncpus', default=int(os.cpu_count()/2), type=int, help ='Number of cpus to use')
parser.add_argument('--clustering', type = str, choices = ['KMedoids', 'Aff_Prop'], help ='Clustering method to use')
parser.add_argument('--rescoring', type = str, nargs='+', choices = ['gnina', 'AD4', 'chemplp', 'rfscorevs', 'LinF9', 'vinardo', 'plp', 'AAScore', 'ECIF', 'SCORCH', 'RTMScore'], help='Rescoring methods to use')

args = parser.parse_args()

if args.pocket == 'reference' and not args.reffile:
    parser.error("Must specify a reference ligand file when --pocket is set to 'reference'")
    
if any(metric in args.clustering for metric in ['RMSD', 'spyRMSD', 'espsim', 'USRCAT']) and not args.clustering:
    parser.error("Must specify a clustering method when --metric is set to 'RMSD', 'spyRMSD', 'espsim' or 'USRCAT'")
    
def run_command(**kwargs):
    w_dir = os.path.dirname(kwargs.get('proteinfile'))
    print('The working directory has been set to:', w_dir)
    create_temp_folder(w_dir+'/temp')
    
    if os.path.isfile(kwargs.get('proteinfile').replace('.pdb', '_pocket.pdb')) == False:
        if kwargs.get('pocket') == 'reference':
            pocket_definition = get_pocket(kwargs.get('reffile'), kwargs.get('proteinfile'), 10)
        if kwargs.get('pocket') == 'RoG':
            pocket_definition = get_pocket_RoG(kwargs.get('reffile'), kwargs.get('proteinfile'))
        elif kwargs.get('pocket') == 'dogsitescorer':
            pocket_definition = binding_site_coordinates_dogsitescorer(kwargs.get('proteinfile'), w_dir, method='volume')
            
    if os.path.isfile(w_dir+'/temp/final_library.sdf') == False:
        prepare_library(kwargs.get('dockinglibrary'), kwargs.get('idcolumn'), kwargs.get('software'), kwargs.get('protonation'), kwargs.get('ncpus'))

    docking_programs = {'GNINA': w_dir+'/temp/gnina/', 'SMINA': w_dir+'/temp/smina/', 'PLANTS': w_dir+'/temp/plants/'}
    for program, file_path in docking_programs.items():
        if os.path.isdir(file_path) == False and program in kwargs.get('docking'):
            docking(w_dir, kwargs.get('proteinfile'), kwargs.get('reffile'), kwargs.get('software'), [program], kwargs.get('exhaustiveness'), kwargs.get('nposes'), kwargs.get('ncpus'), pocket_definition)

    for metric in kwargs.get('metric'):
        if os.path.isfile(w_dir+f'/temp/clustering/{metric}_clustered.sdf') == False:
            print('Loading all poses SDF file...')
            tic = time.perf_counter()
            all_poses = PandasTools.LoadSDF(w_dir+'/temp/allposes.sdf', idName='Pose ID', molColName='Molecule', includeFingerprints=False, strictParsing=True)
            toc = time.perf_counter()
            print(f'Finished loading all poses SDF in {toc-tic:0.4f}!...')

    for metric in kwargs.get('metric'):
        if os.path.isfile(w_dir+f'/temp/clustering/{metric}_clustered.sdf') == False:
            cluster(metric, kwargs.get('clustering'), w_dir, kwargs.get('proteinfile'), all_poses, kwargs.get('ncpus'))
    
    for metric in kwargs.get('metric'):
        rescore_all(w_dir, kwargs.get('proteinfile'), pocket_definition, kwargs.get('software'), w_dir+f'/temp/clustering/{metric}_clustered.sdf', kwargs.get('rescoring'), kwargs.get('ncpus'))

    calculate_EF_single_functions(w_dir, kwargs.get('dockinglibrary'), kwargs.get('metric'))
    
    apply_consensus_methods_combinations(w_dir, kwargs.get('dockinglibrary'), kwargs.get('metric'))
    
run_command(**vars(args))