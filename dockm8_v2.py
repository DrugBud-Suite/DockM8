# Import required libraries and scripts
import yaml
import os
import sys
import re
from pathlib import Path


# Search for 'DockM8' in parent directories
dockm8_path = next((p / 'DockM8' for p in Path(__file__).resolve().parents if (p / 'DockM8').is_dir()), None)
sys.path.append(str(dockm8_path))

from scripts.utilities import printlog
from scripts.docking.docking import DOCKING_PROGRAMS
from scripts.rescoring_functions import RESCORING_FUNCTIONS
from scripts.clustering_metrics import CLUSTERING_METRICS
from scripts.consensus_methods import CONSENSUS_METHODS

class DockM8Error(Exception):
    """Custom Error for DockM8 specific issues."""
    pass

class DockM8Warning(Warning):
    """Custom warning for DockM8 specific issues."""
    pass

# Load YAML configuration function
def load_config(config_path):
    with open(config_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise DockM8Error(f"Error loading configuration file: {e}")

# Assuming the YAML file is named 'config.yaml' and is in the same directory as the script
config = load_config('config.yml')

# Determine software path
software_path = config['general']['software'] if config['general']['software'] != "None" else dockm8_path / "software"
# Check if the software path is valid
if not os.path.isdir(software_path):
    raise DockM8Error(f"DockM8 configuration error : Invalid software path ({software_path}) specified in the configuration file.")

# Set mode
mode = config['general']['mode']
if mode.lower() not in ['single', 'ensemble', 'active_learning']:
    raise DockM8Error("DockM8 configuration error : Invalid mode specified in the configuration file.")

# Set the number of CPUs to use
ncpus = int(config['general']['ncpus']) if int(config['general']['ncpus']) != 0 else int(os.cpu_count()*0.9)

# Set the decoy generation conditions
if config['decoy_generation']['gen_decoys']:
    active_path = config['decoy_generation']['actives']
    if not os.path.isdir(Path(active_path)):
        raise ValueError(f"DockM8 configuration error: Invalid actives path ({active_path}) specified in the configuration file.")
    n_decoys = int(config['decoy_generation']['n_decoys'])
    decoy_model = config['decoy_generation']['decoy_model']
    if decoy_model not in ['DEKOIS', 'DUDE', 'DUDE_P']:
        raise ValueError(f"DockM8 configuration error: Invalid decoy model ({decoy_model}) specified in the configuration file.")

# Set the protein input files
if mode == 'single':
    if len(config['input_files']['receptor(s)']) > 1:
        DockM8Warning("DockM8 configuration warning: Multiple receptor files detected in single mode, only the first file will be used.")
    receptors = config['input_files']['receptor(s)'][0]
else:
    receptors = config['input_files']['receptor(s)']
for receptor in receptors:
    if len(receptor) == 4 and receptor.isallnum():
        printlog(f"PDB ID detected: {receptor}, structure will be downloaded from the PDB.")
    elif len(receptor) == 6:
        printlog(f"Uniprot ID detected: {receptor}, Alphafold structure will be downloaded from the database.")
    else : 
        if not receptor.endswith('.pdb'):
            raise DockM8Error(f"DockM8 configuration error: Invalid receptor file format ({receptor}) specified in the configuration file. Please use .pdb files.")
        if not os.path.isfile(Path(receptor)):
            raise DockM8Error(f"DockM8 configuration error: Invalid receptor path ({receptor}) specified in the configuration file.")

# Set the reference ligand files
if mode == 'single':
    if len(config['input_files']['reference_ligand(s)']) > 1:
        DockM8Warning("DockM8 configuration warning: Multiple reference ligand files detected in single mode, only the first file will be used.")
    reference_ligands = config['input_files']['reference_ligand(s)'][0]
else:
    reference_ligands = config['input_files']['reference_ligand(s)']
for reference_ligand in reference_ligands:
    if not receptor.endswith('.sdf'):
        raise DockM8Error(f"DockM8 configuration error: Invalid reference ligand file format ({receptor}) specified in the configuration file. Please use .sdf files.")
    if not os.path.isfile(Path(receptor)):
        raise DockM8Error(f"DockM8 configuration error: Invalid reference ligand path ({receptor}) specified in the configuration file.")

docking_library = config['input_files']['docking_library']
if not docking_library.endswith('.sdf'):
    raise DockM8Error(f"DockM8 configuration error: Invalid docking library format ({docking_library}) specified in the configuration file. Please use .sdf files.")
if not os.path.isfile(Path(docking_library)):
    raise DockM8Error(f"DockM8 configuration error: Invalid docking library path ({docking_library}) specified in the configuration file.")

# Check if anything in config['protein_preparation'] is incorrectly defined
protein_preparation = config['protein_preparation']
conditions = ['select_best_chain', 'fix_nonstandard_residues', 'fix_missing_residues', 'remove_heteroatoms', 'remove_water', 'protonation']

# Perform validation or error handling based on the defined values
for condition in conditions:
    if not isinstance(protein_preparation.get(condition), bool):
        raise DockM8Error(f"DockM8 configuration error: '{condition}' in 'protein_preparation' section must be a boolean (true/false) value.")

if protein_preparation['add_hydrogens'] is not None or protein_preparation['add_hydrogens'] == 0.0 and protein_preparation['protonation']:
    DockM8Warning("DockM8 configuration warning: 'add_hydrogens' will be ignored as 'protonation' is set to True.")
    
# Check if anything in config['ligand_preparation'] is incorrectly defined
ligand_preparation = config['ligand_preparation']
if ligand_preparation['protonation'] not in ['GypsumDL', 'None']:
    raise DockM8Error("DockM8 configuration error: 'protonation' in 'ligand_preparation' section must be either 'GypsumDL' or 'None'.")
if ligand_preparation['conformers'] not in ['RDKit', 'MMFF', 'GypsumDL']:
    raise DockM8Error("DockM8 configuration error: 'conformers' in 'ligand_preparation' section must be either 'RDKit', 'MMFF' or 'GypsumDL'.")
if not isinstance(ligand_preparation['n_conformers'], int):
    raise DockM8Error("DockM8 configuration error: 'n_conformers' in 'ligand_preparation' section must be an integer value.")

# Check if anything in config['pocket_detection'] is incorrectly defined
pocket_detection = config['pocket_detection']
if pocket_detection['method'] not in ['Reference', 'RoG', 'Dogsitescorer', 'P2Rank', 'Manual']:
    raise DockM8Error("DockM8 configuration error: Invalid pocket detection method specified in the configuration file. Must be either 'Reference', 'RoG', 'Dogsitescorer', 'P2Rank' or 'Manual'.")
if pocket_detection['method'] == 'Reference' or pocket_detection['method'] == 'RoG':
    if not pocket_detection['reference_ligand']:
        raise DockM8Error("DockM8 configuration error: Reference ligand file path is required for 'Reference' and 'RoG' pocket detection methods.")
    if not pocket_detection['reference_ligand'].endswith('.sdf'):
        raise DockM8Error(f"DockM8 configuration error: Invalid reference ligand file format ({pocket_detection['reference_ligand']}) specified in the configuration file. Please use .sdf files.")
    if not os.path.isfile(Path(pocket_detection['reference_ligand'])):
        raise DockM8Error(f"DockM8 configuration error: Invalid reference ligand file path ({pocket_detection['reference_ligand']}) specified in the configuration file.")
if pocket_detection['method'] == 'Reference':
    if not isinstance(pocket_detection['radius'], (float, int)):
        raise DockM8Error("DockM8 configuration error: Pocket detection radius must be a number.")
if pocket_detection['method'] == 'Manual':
    if not pocket_detection['manual_pocket']:
        raise DockM8Error("DockM8 configuration error: Manual pocket definition is required for 'Manual' pocket detection method. Please specify coordinates in the manual_pocket field (Format should be 'center:x,y,z*size:x,y,z').")
    if not re.match(r"center:\d+,\d+,\d+\*size:\d+,\d+,\d+", pocket_detection['manual_pocket']):
        raise DockM8Error("DockM8 configuration error: Invalid manual pocket definition format. Format should be 'center:x,y,z*size:x,y,z' where x y and z are numbers.")

# Check docking configuration
docking = config['docking']
for program in docking['docking_programs']:
    if program not in [DOCKING_PROGRAMS.keys()]:
        raise DockM8Error(f"DockM8 configuration error: Invalid docking program ({program}) specified in the configuration file.")
bust_poses = docking['bust_poses']
if not isinstance(bust_poses, bool):
    raise DockM8Error("DockM8 configuration error: 'bust_poses' in 'docking' section must be a boolean (true/false) value.")
if not isinstance(docking['nposes'], int):
    raise DockM8Error("DockM8 configuration error: 'nposes' in 'docking' section must be an integer value.")
if not isinstance(docking['exhaustiveness'], int):
    raise DockM8Error("DockM8 configuration error: 'exhaustiveness' in 'docking' section must be an integer value.")

# Check pose selection configuration
pose_selection = config['pose_selection']
for method in pose_selection['method']:
    if method not in [CLUSTERING_METRICS.keys()]+['bestpose', 'bestpose_GNINA', 'bestpose_SMINA', 'bestpose_PLANTS', 'bestpose_QVINA2', 'bestpose_QVINAW']+[RESCORING_FUNCTIONS.keys()]:
        raise DockM8Error(f"DockM8 configuration error: Invalid pose selection method ({method}) specified in the configuration file.")
if pose_selection['clustering_method'] not in ['KMedoids', 'Aff_Prop', None]:
    raise DockM8Error("DockM8 configuration error: 'clustering_method' in 'pose_selection' section must be either 'KMedoids' or 'Aff_Prop'.")
if any(method in CLUSTERING_METRICS.keys() for method in pose_selection['method']) and not pose_selection['clustering_method']:
    DockM8Warning("DockM8 configuration warning: 'clustering_method' is not set for clustering metrics, defaulting to 'KMedoids'.")
    pose_selection['clustering_method'] = 'KMedoids'

# Check if anything in config['rescoring'] is incorrectly defined
for method in config['rescoring']:
    if method not in [RESCORING_FUNCTIONS.keys()]:
        raise DockM8Error(f"DockM8 configuration error: Invalid rescoring method ({method}) specified in the configuration file.")

# Check if anything in config['consensus'] is incorrectly defined
if config['consensus'] not in [CONSENSUS_METHODS.keys()]:
    raise DockM8Error(f"DockM8 configuration error: Invalid consensus method ({config['consensus']}) specified in the configuration file. Must be one of {[CONSENSUS_METHODS.keys()]}.")

if mode in ['ensemble', 'active_learing'] and not config['threshold']:
    DockM8Warning(f"DockM8 configuration warning: {mode} mode requires a threshold to be set. Setting to default (1%)")