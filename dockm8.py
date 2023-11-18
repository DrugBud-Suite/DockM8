#Import required libraries and scripts
from scripts.library_preparation import *
from scripts.utilities import *
from scripts.docking_functions import *
from scripts.clustering_functions import *
from scripts.rescoring_functions import *
from scripts.consensus_methods import *
from scripts.performance_calculation import *
from scripts.dogsitescorer import *
from scripts.get_pocket import *
from scripts.postprocessing import *
from scripts.protein_preparation import *
from IPython.display import display
from pathlib import Path
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Parse required arguments')

parser.add_argument('--software', required=True, type=str, help ='Path to the software folder')
parser.add_argument('--mode', type=str, default='single', choices=['single', 'ensemble', 'active_learning'], help ='Specifies the mode: single, ensemble, or active_learning')
parser.add_argument('--split', default=1, type=int, help='Whether to split the docking library into chunks (useful for large libraries)')

parser.add_argument('--receptor', required=True, type=str, nargs='+', help ='Path to the protein file(s) or protein files if using ensemble docking mode')
parser.add_argument('--pocket', required=True, type=str, choices=['reference', 'RoG', 'dogsitescorer'], help ='Method to use for pocket determination')
parser.add_argument('--reffile', type=str, nargs='+', help ='Path to the reference ligand file(s)')
parser.add_argument('--docking_library', required=True, type=str, help ='Path to the docking library file')
parser.add_argument('--idcolumn', required=True, type=str, help ='Column name for the unique identifier')
parser.add_argument('--prepare_proteins', default=1, type=int, help ='Whether or not to add hydrogens to the protein using Protoss (1 for yes, 0 for no)')
parser.add_argument('--protonation', required=True, type=str, choices=['pkasolver', 'GypsumDL', 'None'], help ='Method to use for compound protonation')
parser.add_argument('--docking_programs', required=True, type=str, nargs='+', choices=['GNINA', 'SMINA', 'PLANTS'], help ='Method(s) to use for docking')
parser.add_argument('--clustering_metric', required=True, type=str, nargs='+', choices=['RMSD', 'spyRMSD', 'espsim', 'USRCAT', '3DScore', 'bestpose', 'bestpose_GNINA', 'bestpose_SMINA', 'bestpose_PLANTS'], help ='Method(s) to use for pose clustering')
parser.add_argument('--nposes', default=10, type=int, help ='Number of poses to generate')
parser.add_argument('--exhaustiveness', default=8, type=int, help ='Precision of SMINA/GNINA')
parser.add_argument('--ncpus', default=int(os.cpu_count()/2), type=int, help ='Number of CPUs to use')
parser.add_argument('--clustering_method', type=str, choices=['KMedoids', 'Aff_Prop'], help ='Clustering method to use')
parser.add_argument('--rescoring', type=str, nargs='+', choices=['GNINA_Affinity', 'CNN-Score', 'CNN-Affinity', 'AD4', 'CHEMPLP', 'RFScoreVS', 'LinF9', 'Vinardo', 'PLP', 'AAScore', 'ECIF', 'SCORCH', 'RTMScore', 'NNScore', 'PLECScore', 'KORPL', 'ConvexPLR'], help='Rescoring methods to use')
parser.add_argument('--consensus', type=str, required=True, choices=['method1', 'method2', 'method3', 'method4', 'method5', 'method6', 'method7'], help='Consensus method to use')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for ensemble and active_learning methods')

args = parser.parse_args()

if args.mode == 'ensemble' or args.mode == 'active_learning' and not args.threshold:
    parser.error(f"Must specify a threshold when --mode is set to {args.mode} mode")

if args.pocket == 'reference' or args.pocket == 'RoG' and not args.reffile:
    parser.error(f"Must specify a reference ligand file when --pocket is set to {args.pocket}")
    
if any(metric in args.clustering_metric for metric in ['RMSD', 'spyRMSD', 'espsim', 'USRCAT']) and not args.clustering_method:
    parser.error("Must specify a clustering method when --clustering_metric is set to 'RMSD', 'spyRMSD', 'espsim' or 'USRCAT'")
    
def dockm8(software, receptor, pocket, ref, docking_library, idcolumn, prepare_protein, protonation, docking_programs, clustering_metrics, nposes, exhaustiveness, ncpus, clustering, rescoring, consensus):
    # Set working directory
    w_dir = Path(receptor).parent / Path(receptor).stem
    print('The working directory has been set to:', w_dir)
    (w_dir).mkdir(exist_ok=True)
    # Prepare protein
    if prepare_protein == 1:
        prepared_receptor = prepare_protein_protoss(receptor)
    else:
        prepared_receptor = receptor
    # Determine pocket definition
    if os.path.isfile(str(prepared_receptor).replace('.pdb', '_pocket.pdb')) == False:
        if pocket == 'reference':
            pocket_definition = get_pocket(ref, prepared_receptor, 10)
        if pocket == 'RoG':
            pocket_definition = get_pocket_RoG(ref, prepared_receptor)
        elif pocket == 'dogsitescorer':
            pocket_definition = binding_site_coordinates_dogsitescorer(prepared_receptor, w_dir, method='volume')
    else:
        pocket_definition = calculate_pocket_coordinates_from_pocket_pdb_file((str(prepared_receptor).replace('.pdb', '_pocket.pdb')))
    # Prepare docking library
    if os.path.isfile(w_dir+'/final_library.sdf') == False:
        prepare_library(docking_library, idcolumn, protonation, software, ncpus)
    # Docking
    docking(w_dir, prepared_receptor, pocket_definition, software, docking_programs, exhaustiveness, nposes, ncpus, 'joblib')
    concat_all_poses(w_dir, docking_programs)
    # Clustering
    for metric in clustering_metrics:
        if os.path.isfile(w_dir+f'/clustering/{metric}_clustered.sdf') == False:
            print('Loading all poses SDF file...')
            tic = time.perf_counter()
            all_poses = PandasTools.LoadSDF(w_dir+'/allposes.sdf', idName='Pose ID', molColName='Molecule', includeFingerprints=False, strictParsing=True)
            toc = time.perf_counter()
            print(f'Finished loading all poses SDF in {toc-tic:0.4f}!...')
    for metric in clustering_metrics:
        if os.path.isfile(w_dir+f'/clustering/{metric}_clustered.sdf') == False:
            cluster_pebble(metric, clustering, w_dir, prepared_receptor, all_poses, ncpus)
    # Rescoring
    for metric in clustering_metrics:
        rescore_all(w_dir, prepared_receptor, pocket_definition, software, w_dir+f'/clustering/{metric}_clustered.sdf', rescoring, ncpus)
    # Consensus
    apply_consensus_methods(w_dir, clustering_metrics, consensus, rescoring, standardization_type='min_max')

def run_command(**kwargs):
    # Single mode
    if kwargs.get('mode') == 'single':
        print('DockM8 is running in single mode...')
        
        dockm8(software = kwargs.get('software'),
                receptor = kwargs.get('receptor'), 
                pocket = kwargs.get('pocket'), 
                ref = kwargs.get('reffile'), 
                docking_library = kwargs.get('docking_library'), 
                idcolumn = kwargs.get('idcolumn'), 
                prepare_protein = kwargs.get('prepare_protein'),
                protonation = kwargs.get('protonation'), 
                docking = kwargs.get('docking'), 
                clustering_metrics = kwargs.get('clustering_metric'), 
                nposes = kwargs.get('nposes'), 
                exhaustiveness = kwargs.get('exhaustiveness'), 
                ncpus = kwargs.get('ncpus'), 
                clustering_method = kwargs.get('clustering_method'), 
                rescoring = kwargs.get('rescoring'), 
                consensus = kwargs.get('consensus'))

    # Ensemble mode
    if kwargs.get('mode') == 'ensemble':
        print('DockM8 is running in ensemble mode...')
        
        receptors = kwargs.get('receptor')
        ref_files = kwargs.get('reffile')
        
        receptor_dict = {}
        for i, receptor in enumerate(receptors):
            receptor_dict[receptor] = ref_files[i]
            
        for receptor, ref in receptor_dict.items():
    
            dockm8(software=kwargs.get('software'),
                    receptor = receptor, 
                    pocket = kwargs.get('pocket'), 
                    ref = ref, 
                    docking_library = kwargs.get('docking_library'), 
                    idcolumn = kwargs.get('idcolumn'), 
                    protonation = kwargs.get('protonation'), 
                    docking = kwargs.get('docking'), 
                    clustering_metrics = kwargs.get('clustering_metric'), 
                    nposes = kwargs.get('nposes'), 
                    exhaustiveness = kwargs.get('exhaustiveness'), 
                    ncpus = kwargs.get('ncpus'), 
                    clustering_method = kwargs.get('clustering_method'), 
                    rescoring = kwargs.get('rescoring'), 
                    consensus = kwargs.get('consensus'))
        ensemble_consensus(receptors, kwargs.get('clustering_metric'), kwargs.get('clustering_method'), kwargs.get('threshold'))
        
run_command(**vars(args))