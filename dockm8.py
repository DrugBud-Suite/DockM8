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
parser.add_argument('--mode', type = str, default='single', choices = ['single', 'ensemble', 'active_learning'], help ='Changes mode from classical docking, to ensemble or active learning mode')
parser.add_argument('--receptor', required=True, type=str, nargs='+', help ='Path to protein file or protein files if using ensemble docking mode')
parser.add_argument('--pocket', required=True, type = str, choices = ['reference', 'RoG', 'dogsitescorer'], help ='Method to use for pocket determination')
parser.add_argument('--reffile', type=str, nargs='+', help ='Path to reference ligand file')
parser.add_argument('--dockinglibrary', required=True, type=str, help ='Path to docking library file')
parser.add_argument('--idcolumn', required=True, type=str, help ='Unique identifier column')
parser.add_argument('--protonation', required=True, type = str, choices = ['pkasolver', 'GypsumDL', 'None'], help ='Method to use for compound protonation')
parser.add_argument('--docking', required=True, type = str, nargs='+', choices = ['GNINA', 'SMINA', 'PLANTS'], help ='Method(s) to use for docking')
parser.add_argument('--metric', required=True, type = str, nargs='+', choices = ['RMSD', 'spyRMSD', 'espsim', 'USRCAT', '3DScore', 'bestpose', 'bestpose_GNINA', 'bestpose_SMINA', 'bestpose_PLANTS'], help ='Method(s) to use for pose clustering')
parser.add_argument('--nposes', default=10, type=int, help ='Number of poses')
parser.add_argument('--exhaustiveness', default=8, type = int, help ='Precision of SMINA/GNINA')
parser.add_argument('--ncpus', default=int(os.cpu_count()/2), type=int, help ='Number of cpus to use')
parser.add_argument('--clustering', type = str, choices = ['KMedoids', 'Aff_Prop'], help ='Clustering method to use')
parser.add_argument('--rescoring', type = str, nargs='+', choices = ['GNINA_Affinity', 'CNN-Score', 'CNN-Affinity', 'AD4', 'CHEMPLP', 'RFScoreVS', 'LinF9', 'Vinardo', 'PLP', 'AAScore', 'ECIF', 'SCORCH', 'RTMScore', 'NNScore', 'PLECnn', 'KORPL', 'ConvexPLR'], help='Rescoring methods to use')
parser.add_argument('--consensus', type=str, required=True, choices=['method1', 'method2', 'method3', 'method4', 'method5', 'method6', 'method7'])

args = parser.parse_args()

if args.pocket == 'reference' or args.pocket == 'RoG' and not args.reffile:
    parser.error("Must specify a reference ligand file when --pocket is set to 'reference'")
    
if any(metric in args.clustering for metric in ['RMSD', 'spyRMSD', 'espsim', 'USRCAT']) and not args.clustering:
    parser.error("Must specify a clustering method when --metric is set to 'RMSD', 'spyRMSD', 'espsim' or 'USRCAT'")
    
def run_command(**kwargs):
    if kwargs.get('mode') == 'single':
        print('DockM8 is running in single mode...')
        w_dir = Path(kwargs.get('receptor')).parent / Path(kwargs.get('receptor')).stem
        print('The working directory has been set to:', w_dir)
        (w_dir).mkdir(exist_ok=True)

        if os.path.isfile(str(kwargs.get('receptor')).replace('.pdb', '_pocket.pdb')) == False:
            if kwargs.get('pocket') == 'reference':
                pocket_definition = get_pocket(kwargs.get('reffile'), kwargs.get('receptor'), 10)
            if kwargs.get('pocket') == 'RoG':
                pocket_definition = get_pocket_RoG(kwargs.get('reffile'), kwargs.get('receptor'))
            elif kwargs.get('pocket') == 'dogsitescorer':
                pocket_definition = binding_site_coordinates_dogsitescorer(kwargs.get('receptor'), w_dir, method='volume')
        else:
            pocket_definition = calculate_pocket_coordinates_from_pocket_pdb_file(str(kwargs.get('receptor')).replace('.pdb', '_pocket.pdb'))

        if os.path.isfile(w_dir+'/final_library.sdf') == False:
            prepare_library(kwargs.get('dockinglibrary'), w_dir, kwargs.get('idcolumn'), kwargs.get('protonation'), kwargs.get('software'), kwargs.get('ncpus'))

        docking_programs = ['GNINA', 'SMINA', 'PLANTS', 'QVINA2', 'QVINAW']
        for program in docking_programs:
            if program in kwargs.get('docking'):
                docking(w_dir, kwargs.get('receptor'), pocket_definition, [program], kwargs.get('exhaustiveness'), kwargs.get('nposes'), kwargs.get('ncpus'))

        concat_all_poses(w_dir, docking_programs)
        
        for metric in kwargs.get('metric'):
            if os.path.isfile(w_dir+f'/clustering/{metric}_clustered.sdf') == False:
                print('Loading all poses SDF file...')
                tic = time.perf_counter()
                all_poses = PandasTools.LoadSDF(w_dir+'/allposes.sdf', idName='Pose ID', molColName='Molecule', includeFingerprints=False, strictParsing=True)
                toc = time.perf_counter()
                print(f'Finished loading all poses SDF in {toc-tic:0.4f}!...')

        for metric in kwargs.get('metric'):
            if os.path.isfile(w_dir+f'/clustering/{metric}_clustered.sdf') == False:
                cluster_pebble(metric, kwargs.get('clustering'), w_dir, kwargs.get('receptor'), all_poses, kwargs.get('ncpus'))
        
        for metric in kwargs.get('metric'):
            rescore_all(w_dir, kwargs.get('receptor'), pocket_definition, w_dir+f'/clustering/{metric}_clustered.sdf', kwargs.get('rescoring'), kwargs.get('ncpus'))

        apply_consensus_methods(w_dir, kwargs.get('dockinglibrary'), kwargs.get('metric'))
        
    if kwargs.get('mode') == 'ensemble':
        print('DockM8 is running in ensemble mode...')
        
        receptors = kwargs.get('receptor')
        ref_files = kwargs.get('reffile')
        
        receptor_dict = {}
        for i, receptor in enumerate(receptors):
            receptor_dict[receptor] = ref_files[i]
            
        for receptor, ref in receptor_dict.items():
    
            w_dir = Path(receptor).parent / Path(receptor).stem
            print('The working directory has been set to:', w_dir)
            (w_dir).mkdir(exist_ok=True)

            if os.path.isfile(str(receptor).replace('.pdb', '_pocket.pdb')) == False:
                if kwargs.get('pocket') == 'reference':
                    pocket_definition = get_pocket(ref, receptor, 10)
                if kwargs.get('pocket') == 'RoG':
                    pocket_definition = get_pocket_RoG(ref, receptor)
                elif kwargs.get('pocket') == 'dogsitescorer':
                    pocket_definition = binding_site_coordinates_dogsitescorer(receptor, w_dir, method='volume')
            else:
                pocket_definition = calculate_pocket_coordinates_from_pocket_pdb_file((str(receptor).replace('.pdb', '_pocket.pdb')))

            if os.path.isfile(w_dir+'/final_library.sdf') == False:
                prepare_library(kwargs.get('dockinglibrary'), kwargs.get('idcolumn'), kwargs.get('protonation'), kwargs.get('ncpus'))

            docking_programs = ['GNINA', 'SMINA', 'PLANTS', 'QVINA2', 'QVINAW']
            for program in docking_programs:
                if program in kwargs.get('docking'):
                    docking(w_dir, receptor, pocket_definition, [program], kwargs.get('exhaustiveness'), kwargs.get('nposes'), kwargs.get('ncpus'))

            concat_all_poses(w_dir, docking_programs)
            
            for metric in kwargs.get('metric'):
                if os.path.isfile(w_dir+f'/clustering/{metric}_clustered.sdf') == False:
                    print('Loading all poses SDF file...')
                    tic = time.perf_counter()
                    all_poses = PandasTools.LoadSDF(w_dir+'/allposes.sdf', idName='Pose ID', molColName='Molecule', includeFingerprints=False, strictParsing=True)
                    toc = time.perf_counter()
                    print(f'Finished loading all poses SDF in {toc-tic:0.4f}!...')

            for metric in kwargs.get('metric'):
                if os.path.isfile(w_dir+f'/clustering/{metric}_clustered.sdf') == False:
                    cluster_pebble(metric, kwargs.get('clustering'), w_dir, receptor, all_poses, kwargs.get('ncpus'))
            
            for metric in kwargs.get('metric'):
                rescore_all(w_dir, receptor, pocket_definition, w_dir+f'/clustering/{metric}_clustered.sdf', kwargs.get('rescoring'), kwargs.get('ncpus'))

            #apply_consensus_methods_combinations(w_dir, kwargs.get('dockinglibrary'), kwargs.get('metric'))
    
run_command(**vars(args))