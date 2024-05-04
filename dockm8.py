# Import required libraries and scripts
import argparse
import math
import os
import sys
import warnings
from pathlib import Path
import time

from rdkit.Chem import PandasTools

cwd = Path.cwd()
dockm8_path = next((path for path in cwd.parents if path.name == "DockM8"), None)
sys.path.append(str(dockm8_path))

# Import modules for docking, scoring, protein and ligand preparation, etc.
from scripts.clustering_functions import *
from scripts.consensus_methods import *
from scripts.docking.docking import *
from scripts.library_preparation.main import prepare_library
from scripts.performance_calculation import *
from scripts.pocket_finding.main import pocket_finder
from scripts.postprocessing import *
from scripts.protein_preparation import *
from scripts.rescoring_functions import *
from scripts.utilities import *
from software.DeepCoy.generate_decoys import generate_decoys

# Suppress warnings to clean up output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize argument parser to handle command line arguments
parser = argparse.ArgumentParser(description='Parse required arguments')

# Define command line arguments for the script
parser.add_argument('--software', required=True, type=str, help ='Path to the software folder')
parser.add_argument('--mode', type=str, default='single', choices=['Single', 'Ensemble', 'active_learning'], help ='Specifies the mode: single, ensemble, or active_learning')

parser.add_argument('--gen_decoys', default=False, type=str2bool, help ='Whether or not to generate decoys using DeepCoy')
parser.add_argument('--decoy_model', default='DUDE', type=str, choices=['DUDE', 'DEKOIS', 'DUDE_P'], help ='Model to use for decoy generation')
parser.add_argument('--n_decoys', default=20, type=int, help ='Number of decoys to generate')
parser.add_argument('--actives', default=None, type=str, help ='Path to the list of active compounds .sdf file')

parser.add_argument('--receptor', required=True, type=str, nargs='+', help ='Path to the protein file(s) or protein files if using ensemble docking mode')
parser.add_argument('--pocket', type=str, help ='Method to use for pocket determination. Choices are "Reference", "RoG", "Dogsitescorer", "p2rank" or manual pocket coordinates in the format "center:x,y,z*size:x,y,z"')
parser.add_argument('--reffile', type=str, nargs='+', help ='Path to the reference ligand file(s)')
parser.add_argument('--docking_library', required=True, type=str, help ='Path to the docking library .sdf file')
parser.add_argument('--idcolumn', required=True, type=str, help ='Column name for the unique identifier')
parser.add_argument('--prepare_proteins', default=True, type=str2bool, help ='Whether or not to add hydrogens to the protein using Protoss (True for yes, False for no)')
parser.add_argument('--conformers', default='RDKit', type=str, choices=['RDKit', 'MMFF', 'GypsumDL'], help ='Method to use for conformer generation (RDKit and MMFF are equivalent). Choices are "GypsumDL", "RDKit" or "MMFF"')
parser.add_argument('--protonation', required=True, type=str, choices=['GypsumDL', 'None'], help ='Method to use for compound protonation. Choices are "GypsumDL" or "None"')
parser.add_argument('--docking_programs', required=True, type=str, nargs='+', choices=DOCKING_PROGRAMS, help ='Method(s) to use for docking. Choices are "SMINA", "GNINA", "PLANTS", "QVINA2", "QVINAW" or any combination thereof, separated by spaces')
parser.add_argument('--bust_poses', default=False, type=str2bool, help ='Whether or not to remove problematic poses with PoseBusters (True for yes, False for no)')
parser.add_argument('--pose_selection', required=True, type=str, nargs='+', choices=list(CLUSTERING_METRICS.keys())+['bestpose', 'bestpose_GNINA', 'bestpose_SMINA', 'bestpose_PLANTS', 'bestpose_QVINA2', 'bestpose_QVINAW']+list(RESCORING_FUNCTIONS.keys()), help ='Method(s) to use for pose clustering')
parser.add_argument('--nposes', default=10, type=int, help ='Number of poses to generate')
parser.add_argument('--exhaustiveness', default=8, type=int, help ='Precision of SMINA/GNINA')
parser.add_argument('--ncpus', default=int(os.cpu_count()*0.9), type=int, help ='Number of CPUs to use')
parser.add_argument('--clustering_method', type=str, choices=['KMedoids', 'Aff_Prop', 'None'], help ='Clustering method to use')
parser.add_argument('--rescoring', type=str, nargs='+', choices=list(RESCORING_FUNCTIONS.keys()), help='Rescoring methods to use')
parser.add_argument('--consensus', type=str, required=True, choices=['ECR_best', 'ECR_avg', 'avg_ECR', 'RbR', 'RbV', 'Zscore_best', 'Zscore_avg'], help='Consensus method to use')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for ensemble and active_learning methods')

# Parse arguments from command line
args = parser.parse_args()

# Adjust the receptor argument based on the mode
if args.mode == 'ensemble':
    # Treat --receptor as a list
    receptors = args.receptor
    reffile = args.reffile
else:
    # Treat --receptor as a single file
    receptors = [args.receptor[0]]
    if args.reffile:
        reffile = [args.reffile[0]]

# Validate arguments based on the specified mode
if args.mode == 'ensemble' or args.mode == 'active_learning' and not args.threshold:
    parser.error(f"Must specify a threshold when --mode is set to {args.mode} mode")

if (args.pocket == 'Reference' or args.pocket == 'RoG') and not args.reffile:
    parser.error(f"Must specify a reference ligand file when --pocket is set to {args.pocket}")
    
if any(metric in args.pose_selection for metric in CLUSTERING_METRICS.keys()) and (args.clustering_method == None or args.clustering_method == 'None'):
    parser.error("Must specify a clustering method when --pose_selection is set to 'RMSD', 'spyRMSD', 'espsim' or 'USRCAT'")

if args.gen_decoys and not args.decoy_model:
    parser.error("Must specify a decoy model when --gen_decoys is set to True")

if args.gen_decoys and not args.n_decoys:
    parser.error("Must specify the number of decoys to generate when --gen_decoys is set to True")

if args.gen_decoys and not args.actives:
    parser.error("Must specify the path to the actives file when --gen_decoys is set to True")

if args.gen_decoys and len(args.rescoring) > 8:
    # Warn about the large number of combinations that might be tried during optimization
    possibilites = math.factorial(len(args.rescoring))*len(args.clustering_metric)*len(args.docking_programs)*7
    print(f"WARNING : At least {possibilites} possible combinations will be tried for optimization, this may take a while.")

# Validate pose selection method against docking programs
for program in DOCKING_PROGRAMS:
    if f"bestpose_{program}" in args.pose_selection and program not in args.docking_programs:
        parser.error(f"Must specify {program} in --docking_programs when --pose_selection is set to bestpose_{program}")

# Main function to manage docking process
def dockm8(software, receptor, pocket, ref, docking_library, idcolumn, prepare_proteins, conformers, protonation, docking_programs, bust_poses, pose_selection, nposes, exhaustiveness, ncpus, clustering_method, rescoring, consensus):
    # Set working directory based on the receptor file
    w_dir = Path(receptor).parent / Path(receptor).stem
    print('The working directory has been set to:', w_dir)
    (w_dir).mkdir(exist_ok=True)
    
    # Prepare the protein for docking (e.g., adding hydrogens)
    if prepare_proteins:
        prepared_receptor = Path(prepare_protein_protoss(receptor))
    else:
        prepared_receptor = Path(receptor)
    
    # Determine the docking pocket
    pocket_definition = pocket_finder(pocket, w_dir, prepared_receptor, ref, 10)
    
    print("The pocket coordinates are:", pocket_definition)
        
    # Prepare the docking library if not already prepared
    if not os.path.isfile(w_dir / 'final_library.sdf'):
        prepare_library(docking_library, w_dir, idcolumn, protonation, conformers, software, ncpus)
    
    # Perform the docking operation
    docking(w_dir, prepared_receptor, pocket_definition, software, docking_programs, exhaustiveness, nposes, ncpus, 'concurrent_process')
    
    # Concatenate all poses into a single file
    concat_all_poses(w_dir, docking_programs, prepared_receptor, ncpus, bust_poses)
    
    # Load all poses from SDF file and perform clustering
    print('Loading all poses SDF file...')
    tic = time.perf_counter()
    all_poses = PandasTools.LoadSDF(str(w_dir / 'allposes.sdf'), idName='Pose ID', molColName='Molecule', includeFingerprints=False, strictParsing=True)
    toc = time.perf_counter()
    print(f'Finished loading all poses SDF in {toc-tic:0.4f}!')
    for method in pose_selection:
        if not os.path.isfile(w_dir / f'clustering/{method}_clustered.sdf'):
            select_poses(method, clustering_method, w_dir, prepared_receptor, pocket_definition, software, all_poses, ncpus)
    
    # Rescore poses for each selection method
    for method in pose_selection:
        rescore_poses(w_dir, prepared_receptor, pocket_definition, software, w_dir / 'clustering' / f'{method}_clustered.sdf', rescoring, ncpus)
    
    # Apply consensus methods to the poses
    for method in pose_selection:
        apply_consensus_methods(w_dir, method, consensus, rescoring, standardization_type='min_max')

def run_command(**kwargs):
    # Run DockM8 in decoy mode
    if kwargs.get('gen_decoys'):
        if kwargs.get('mode') == 'Single':
            print('DockM8 is running in single mode...')
            print('DockM8 is generating decoys...')
            # Generate decoys
            output_library = generate_decoys(Path(kwargs.get('actives')), kwargs.get('n_decoys'), kwargs.get('decoy_model'), kwargs.get('software'))
            # Run DockM8 on decoy library
            dockm8(software = Path(kwargs.get('software')),
                    receptor = (kwargs.get('receptor'))[0], 
                    pocket = kwargs.get('pocket'), 
                    ref = (kwargs.get('reffile'))[0] if kwargs.get('reffile') else None, 
                    docking_library = output_library, 
                    idcolumn = kwargs.get('idcolumn'), 
                    prepare_proteins = kwargs.get('prepare_proteins'),
                    conformers=kwargs.get('conformers'),
                    protonation = kwargs.get('protonation'), 
                    docking_programs = kwargs.get('docking_programs'),
                    bust_poses = kwargs.get('bust_poses'), 
                    pose_selection = kwargs.get('pose_selection'), 
                    nposes = kwargs.get('nposes'), 
                    exhaustiveness = kwargs.get('exhaustiveness'), 
                    ncpus = kwargs.get('ncpus'), 
                    clustering_method = kwargs.get('clustering_method'), 
                    rescoring = kwargs.get('rescoring'), 
                    consensus = None)
            performance = calculate_performance(output_library.parent, 
                                    output_library,
                                    [10, 5, 2, 1, 0.5])
            #Determine optimal conditions
            optimal_conditions = performance.sort_values(by='EF1', ascending=False).iloc[0].to_dict()
            if optimal_conditions['clustering'] == 'bestpose':
                docking_programs = kwargs.get('docking_programs')
            if '_' in optimal_conditions['clustering']:
                docking_programs = list(optimal_conditions['clustering'].split('_')[1])
            else:
                docking_programs = kwargs.get('docking_programs')
            optimal_rescoring_functions = list(optimal_conditions['scoring'].split('_'))
            # Save optimal conditions to a file
            with open((kwargs.get('receptor'))[0].parent / 'DeepCoy' / 'optimal_conditions.txt', 'w') as file:
                file.write(str(optimal_conditions))
            # Run DockM8 on the test library using the optimal conditions
            dockm8(software = Path(kwargs.get('software')),
                    receptor = (kwargs.get('receptor'))[0], 
                    pocket = kwargs.get('pocket'), 
                    ref = (kwargs.get('reffile'))[0] if kwargs.get('reffile') else None, 
                    docking_library = kwargs.get('docking_library'), 
                    idcolumn = kwargs.get('idcolumn'), 
                    prepare_proteins = kwargs.get('prepare_proteins'),
                    conformers=kwargs.get('conformers'),
                    protonation = kwargs.get('protonation'), 
                    docking_programs = docking_programs, 
                    bust_poses = kwargs.get('bust_poses'), 
                    pose_selection = list(optimal_conditions['clustering']), 
                    nposes = kwargs.get('nposes'), 
                    exhaustiveness = kwargs.get('exhaustiveness'), 
                    ncpus = kwargs.get('ncpus'), 
                    clustering_method = kwargs.get('clustering_method'), 
                    rescoring = optimal_rescoring_functions, 
                    consensus = optimal_conditions['consensus'])
        if kwargs.get('mode') == 'Ensemble':
            print('DockM8 is running in ensemble mode...')
            print('DockM8 is generating decoys...')
            # Generate decoys
            output_library = generate_decoys(Path(kwargs.get('actives')), kwargs.get('n_decoys'), kwargs.get('decoy_model'), kwargs.get('software'))
            # Run DockM8 on the decoys library
            dockm8(software = Path(kwargs.get('software')),
                    receptor = (kwargs.get('receptor'))[0], 
                    pocket = kwargs.get('pocket'), 
                    ref = (kwargs.get('reffile'))[0] if kwargs.get('reffile') else None, 
                    docking_library = output_library, 
                    idcolumn = kwargs.get('idcolumn'), 
                    prepare_proteins = kwargs.get('prepare_proteins'),
                    conformers=kwargs.get('conformers'),
                    protonation = kwargs.get('protonation'), 
                    docking_programs = kwargs.get('docking_programs'), 
                    bust_poses = kwargs.get('bust_poses'), 
                    pose_selection = kwargs.get('pose_selection'), 
                    nposes = kwargs.get('nposes'), 
                    exhaustiveness = kwargs.get('exhaustiveness'), 
                    ncpus = kwargs.get('ncpus'), 
                    clustering_method = kwargs.get('clustering_method'), 
                    rescoring = kwargs.get('rescoring'), 
                    consensus = None)
            performance = calculate_performance(output_library.parent, 
                                    output_library,
                                    [10, 5, 2, 1, 0.5])
            #Determine optimal conditions
            optimal_conditions = performance.sort_values(by='EF1', ascending=False).iloc[0].to_dict()
            if optimal_conditions['clustering'] == 'bestpose':
                docking_programs = kwargs.get('docking_programs')
            if '_' in optimal_conditions['clustering']:
                docking_programs = list(optimal_conditions['clustering'].split('_')[1])
            else:
                docking_programs = kwargs.get('docking_programs')
            optimal_rescoring_functions = list(optimal_conditions['scoring'].split('_'))
            # Save optimal conditions to a file
            with open((kwargs.get('receptor'))[0].parent / 'DeepCoy' / 'optimal_conditions.txt', 'w') as file:
                file.write(str(optimal_conditions))
            # Generate target and reference ligand dictionnary
            receptors = kwargs.get('receptor')
            ref_files = kwargs.get('reffile')
            receptor_dict = {}
            for i, receptor in enumerate(receptors):
                receptor_dict[receptor] = ref_files[i]
            # Run DockM8 on the test library in ensemble mode
            for receptor, ref in receptor_dict.items():
                dockm8(software = Path(kwargs.get('software')),
                        receptor = receptor, 
                        pocket = kwargs.get('pocket'), 
                        ref = ref, 
                        docking_library = kwargs.get('docking_library'), 
                        idcolumn = kwargs.get('idcolumn'), 
                        prepare_proteins = kwargs.get('prepare_proteins'),
                        conformers=kwargs.get('conformers'), 
                        protonation = kwargs.get('protonation'), 
                        docking_programs = docking_programs, 
                        bust_poses = kwargs.get('bust_poses'), 
                        pose_selection = list(optimal_conditions['clustering']), 
                        nposes = kwargs.get('nposes'), 
                        exhaustiveness = kwargs.get('exhaustiveness'), 
                        ncpus = kwargs.get('ncpus'), 
                        clustering_method = kwargs.get('clustering_method'), 
                        rescoring = optimal_rescoring_functions, 
                        consensus = optimal_conditions['consensus'])
            ensemble_consensus(receptors, optimal_conditions['clustering'], optimal_conditions['consensus'], kwargs.get('threshold'))
    else:
        # Run DockM8 in single mode
        if kwargs.get('mode') == 'Single':
            print('DockM8 is running in single mode...')
            dockm8(software = Path(kwargs.get('software')),
                    receptor = (kwargs.get('receptor'))[0], 
                    pocket = kwargs.get('pocket'), 
                    ref = (kwargs.get('reffile'))[0] if kwargs.get('reffile') else None, 
                    docking_library = kwargs.get('docking_library'), 
                    idcolumn = kwargs.get('idcolumn'), 
                    prepare_proteins = kwargs.get('prepare_proteins'),
                    conformers=kwargs.get('conformers'),
                    protonation = kwargs.get('protonation'), 
                    docking_programs = kwargs.get('docking_programs'), 
                    bust_poses = kwargs.get('bust_poses'), 
                    pose_selection = kwargs.get('pose_selection'), 
                    nposes = kwargs.get('nposes'), 
                    exhaustiveness = kwargs.get('exhaustiveness'), 
                    ncpus = kwargs.get('ncpus'), 
                    clustering_method = kwargs.get('clustering_method'), 
                    rescoring = kwargs.get('rescoring'), 
                    consensus = kwargs.get('consensus'))
        # Ensemble mode
        if kwargs.get('mode') == 'Ensemble':
            print('DockM8 is running in ensemble mode...')
            # Generate target and reference ligand dictionnary
            receptors = kwargs.get('receptor')
            ref_files = kwargs.get('reffile')
            receptor_dict = {}
            for i, receptor in enumerate(receptors):
                receptor_dict[receptor] = ref_files[i]
            # Run DockM8 in ensemble mode
            for receptor, ref in receptor_dict.items():
                dockm8(software = Path(kwargs.get('software')),
                        receptor = receptor, 
                        pocket = kwargs.get('pocket'), 
                        ref = ref, 
                        docking_library = kwargs.get('docking_library'), 
                        idcolumn = kwargs.get('idcolumn'), 
                        prepare_proteins = kwargs.get('prepare_proteins'),
                        conformers=kwargs.get('conformers'), 
                        protonation = kwargs.get('protonation'), 
                        docking_programs = kwargs.get('docking_programs'), 
                        bust_poses = kwargs.get('bust_poses'), 
                        pose_selection = kwargs.get('pose_selection'), 
                        nposes = kwargs.get('nposes'), 
                        exhaustiveness = kwargs.get('exhaustiveness'), 
                        ncpus = kwargs.get('ncpus'), 
                        clustering_method = kwargs.get('clustering_method'), 
                        rescoring = kwargs.get('rescoring'), 
                        consensus = kwargs.get('consensus'))
            ensemble_consensus(receptors, kwargs.get('pose_selection'), kwargs.get('consensus'), kwargs.get('threshold'))

run_command(**vars(args))