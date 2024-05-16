# Import necessary libraries
import itertools
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import streamlit as st

from scripts.clustering_metrics import CLUSTERING_METRICS
from scripts.consensus_methods import CONSENSUS_METHODS
from scripts.docking.docking import DOCKING_PROGRAMS
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS
from scripts.pocket_finding.pocket_finding import POCKET_DETECTION_OPTIONS

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(page_title="DockM8",
                   page_icon="./media/DockM8_logo.png",
                   layout="wide")
# Sidebar
st.sidebar.image(image="./media/DockM8_white_horizontal.png", width=200)
st.sidebar.title("DockM8")
st.sidebar.subheader("Open-source consensus docking for everyone")
st.sidebar.link_button("Github", url="https://github.com/DrugBud-Suite/DockM8")
st.sidebar.link_button("Visit Website",
                       url="https://drugbud-suite.github.io/dockm8-web/")
st.sidebar.link_button("Publication", url="https://doi.org/your-doi")
st.sidebar.link_button("Zenodo repository", url="https://doi.org/your-doi")

# Logo
st.columns(3)[1].image(image="./media/DockM8_white_vertical.png", width=400)

# Setup config dictionnary
config = {}

# Setup
CWD = os.getcwd()
st.header("Setup", divider="orange")
mode = st.selectbox(
    label="Which mode do you want to run DockM8 in?",
    key=0,
    options=("Single", "Ensemble"),
    help=
    "Single Docking: DockM8 will dock each ligand in the library to a single receptor. "
    +
    "Ensemble Docking: DockM8 will dock each ligand in the library to all specified receptors and then combine the results to create an ensemble consensus.",
)
if mode != "Single":
    threshold = st.slider(
        label="Threshold for ensemble consensus (in %)",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        value=1.0,
        help=
        "Threshold for ensemble consensus (in %). DockM8 will only consider a ligand as a consensus hit if it is a top scorer for all the receptors.",
    )
else:
    threshold = None
n_cpus = st.slider(
    "Number of CPUs",
    min_value=1,
    max_value=os.cpu_count(),
    step=1,
    value=int(os.cpu_count() * 0.9),
    help="Number of CPUs to use for calculations",
)
software = st.text_input(
    "Choose a software directory",
    value=CWD + "/software",
    help=
    "Type the directory containing the software folder: For example: /home/user/Dockm8/software",
)

gen_decoys = st.toggle(
    label="Generate decoys",
    value=False,
    help=
    "Generate decoys for the active ligands and determine optimal DockM8 conditions",
)

if mode == "Single":
    receptor_value = CWD + "/tests/test_files/1fvv_p.pdb"
    reference_value = CWD + "/tests/test_files/1fvv_l.sdf"
    library_value = CWD + "/tests/test_files/library.sdf"
if mode == "Ensemble":
    receptor_value = (CWD + "/tests/test_files/4kd1_p.pdb, " + CWD +
                      "/tests/test_files/1fvv_p.pdb")
    reference_value = (CWD + "/tests/test_files/4kd1_l.sdf, " + CWD +
                       "/tests/test_files/1fvv_l.sdf")
    library_value = CWD + "/tests/test_files/library.sdf"

if gen_decoys:
    st.subheader("Decoy generation", divider="orange")
    # Active ligands
    actives = Path(
        st.text_input(
            label="Enter the path to the active ligands file (.sdf format)",
            value=CWD + "/tests/test_files/CDK2_actives.sdf",
            help="Choose an active ligands file (.sdf format)",
        ))
    if not actives.is_file():
        st.error(f"Invalid file path: {actives}")
    # Number of decoys
    n_decoys = st.slider(
        label="Number of decoys",
        min_value=1,
        max_value=100,
        step=1,
        value=10,
        help="Number of decoys to generate for each active ligand",
    )
    # Decoy generation program
    decoy_model = st.selectbox(
        label="Which decoy generation model do you want to use?",
        options=("DUD-E", "DEKOIS", "DUD-E_phosphorus"),
        help="Select which Deepcoy decoy generation model you want to use ",
    )
else:
    actives = n_decoys = decoy_model = None

# Receptor(s)
st.header("Receptor(s)", divider="orange")
receptors = st.text_input(
    label=
    "File path(s) of one or more multiple receptor files (.pdb format), separated by commas",
    help=
    "Choose one or multiple receptor files (.pdb format). Ensure there are no spaces in the file or directory names",
    value=receptor_value,
    placeholder="Enter path(s) here",
)
receptors = [Path(receptor.strip()) for receptor in receptors.split(",")]
# Receptor files validation
for file in receptors:
    if not Path(file).is_file():
        st.error(f"Invalid file path: {file}")

# Prepare receptor
st.header("Receptor Preparation", divider="orange")
select_best_chain = st.toggle(label="AutoSelect best chain",
                              key="select_best_chain",
                              value=False)
fix_nonstandard_residues = st.toggle(label="Fix non standard residues",
                                     key="fix_nonstandard_residues",
                                     value=True)
fix_missing_residues = st.toggle(label="Fix mising residues",
                                 key="fix_missing_residues",
                                 value=True)
remove_heteroatoms = st.toggle(label="Remove ligands and heteroatoms",
                               key="remove_heteroatoms",
                               value=True)
remove_water = st.toggle(label="Remove water", key="remove_water", value=True)
st.subheader("Receptor Protonation", divider="orange")
protonation = st.toggle(
    label=
    "Automatically protonate receptor using Protoss (untoggle to choose a specific pH)",
    value=True,
    help=
    "Choose whether or not to use Protoss Web service to protonate the protein structure",
)
if not protonation:
    add_hydrogens = st.number_input(
        label="Add hydrogens with PDB Fixer at pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
    )
else:
    add_hydrogens = None

# Pocket finding
st.header("Binding Pocket definition", divider="orange")
pocket_mode = st.selectbox(
    label="How should the pocket be defined?",
    options=POCKET_DETECTION_OPTIONS,
    help=
    "Reference Ligand: DockM8 will use the reference ligand to define the pocket. "
    + "RoG: DockM8 will use the reference ligand radius of gyration. " +
    "DogSiteScorer: DockM8 will use the DogSiteScorer pocket finding algorithm to define the pocket."
    +
    "P2Rank: DockM8 will use the P2Rank pocket finding algorithm to define the pocket."
    + "Manual: Define your own pocket center and size coordinates.",
)
if pocket_mode in ("Reference", "RoG"):
    pocket_radius = st.number_input("Binding Site Radius",
                                    min_value=0.0,
                                    value=10.0,
                                    step=0.1)
    reference_files = st.text_input(
        label=
        "File path(s) of one or more multiple reference ligand files (.sdf format), separated by commas",
        help="Choose one or multiple reference ligand files (.pdb format)",
        value=reference_value,
    )
    reference_files = [
        Path(file.strip()) for file in reference_files.split(",")
    ]
    # Reference files validation
    for file in reference_files:
        if not Path(file).is_file():
            st.error(f"Invalid file path: {file}")
    x_center = y_center = z_center = x_size = y_size = z_size = manual_pocket = None
# Custom pocket
elif pocket_mode == "Manual" and mode == "Single":
    col1, col2, col3 = st.columns(3)
    x_center = col1.number_input(
        label="X Center",
        value=0.0,
        help="Enter the X coordinate of the pocket center")
    y_center = col2.number_input(
        label="Y Center",
        value=0.0,
        help="Enter the Y coordinate of the pocket center")
    z_center = col3.number_input(
        label="Z Center",
        value=0.0,
        help="Enter the Z coordinate of the pocket center")
    x_size = col1.number_input(
        label="X Size",
        value=20.0,
        help="Enter the size of the pocket in the X direction (in Angstroms)",
    )
    y_size = col2.number_input(
        label="Y Size",
        value=20.0,
        help="Enter the size of the pocket in the Y direction (in Angstroms)",
    )
    z_size = col3.number_input(
        label="Z Size",
        value=20.0,
        help="Enter the size of the pocket in the Z direction (in Angstroms)",
    )
    manual_pocket = (
        f"center:{x_center},{y_center},{z_center}*size:{x_size},{y_size},{z_size}"
    )
    pocket_radius = reference_files = None
elif pocket_mode == "Manual" and mode != "Single":
    st.error(
        "Manual pocket definition does not currently work in ensemble mode, please change the pocket definition mode"
    )
else:
    pocket_radius = reference_files = x_center = y_center = z_center = x_size = y_size = z_size = manual_pocket = None

# Ligand library
st.header("Ligands", divider="orange")
docking_library = st.text_input(
    label="Entre the path to the ligand library file (.sdf format)",
    value=library_value,
    help="Choose a ligand library file (.sdf format)",
)
if not Path(docking_library).is_file():
    st.error(f"Invalid file path: {docking_library}")

# Ligand protonation
st.subheader("Ligand protonation", divider="orange")
ligand_protonation = st.selectbox(
    label="How should the ligands be protonated?",
    options=("None", "GypsumDL"),
    index=1,
    help="None: No protonation " +
    "Gypsum-DL: DockM8 will use Gypsum-DL to protonate the ligands",
)

# Ligand conformers
st.subheader("Ligand conformers", divider="orange")
ligand_conformers = st.selectbox(
    label="How should the conformers be generated?",
    options=["MMFF", "GypsumDL"],
    index=1,
    help="MMFF: DockM8 will use MMFF to prepare the ligand 3D conformers. " +
    "GypsumDL: DockM8 will use Gypsum-DL to prepare the ligand 3D conformers.",
)
n_conformers = st.number_input("Number of conformers to generate.",
                               min_value=1,
                               max_value=100,
                               step=1)

# Docking programs
st.header("Docking programs", divider="orange")
docking_programs = st.multiselect(
    label="Choose the docking programs you want to use",
    default=["GNINA"],
    options=DOCKING_PROGRAMS,
    help=
    "Choose the docking programs you want to use, mutliple selection is allowed",
)

if "PLANTS" in docking_programs and not os.path.exists(
        "/path/to/software/PLANTS"):
    st.warning(
        "PLANTS was not found in the software folder, please visit http://www.tcd.uni-konstanz.de/research/plants.php"
    )

# Number of poses
n_poses = st.slider(
    label="Number of poses",
    min_value=1,
    max_value=100,
    step=5,
    value=10,
    help="Number of poses to generate for each ligand",
)

# Exhaustiveness
exhaustiveness = st.select_slider(
    label="Exhaustiveness",
    options=[1, 2, 4, 8, 16, 32],
    value=8,
    help=
    "Exhaustiveness of the docking, only applies to GNINA, SMINA, QVINA2 and QVINAW. Higher values can significantly increase the runtime.",
)

# Post docking
st.header("Docking postprocessing", divider="orange")
clash_cutoff_toggle = st.toggle(label="Remove poses with clashes",
                                value=True,
                                help="Remove poses with clashes",
                                key="clash_cutoff_toggle")
if clash_cutoff_toggle:
    clash_cutoff = st.number_input(
        label=
        "Remove poses with more than x clashes: (setting too low will remove to many poses, use default if unsure)",
        min_value=0,
        max_value=100,
        value=5,
        step=1)
else:
    clash_cutoff = None

strain_cutoff_toggle = st.toggle(label="Remove poses with high strain",
                                 value=True,
                                 help="Remove poses with high strain",
                                 key="strain_cutoff_toggle")
if strain_cutoff_toggle:
    strain_cutoff = st.number_input(
        label=
        "Remove poses with higher than x strain energy (kcal/mol): (setting too low will remove to many poses, use default if unsure)",
        min_value=100,
        max_value=100000,
        value=5000,
        step=100)
else:
    strain_cutoff = None

bust_poses = st.toggle(
    label="Bust poses using PoseBusters",
    value=True,
    help=
    "Bust poses using PoseBusters : Will remove any poses with clashes, non-flat aromatic rings etc. WARNING may take a long time to run",
    key="bust_poses_toggle")

# Pose selection
st.header("Pose Selection", divider="orange")
pose_selection = st.multiselect(
    label="Choose the pose selection method you want to use",
    default=["KORP-PL"],
    options=list(CLUSTERING_METRICS.keys()) + [
        "bestpose",
        "bestpose_GNINA",
        "bestpose_SMINA",
        "bestpose_PLANTS",
        "bestpose_QVINA2",
        "bestpose_QVINAW",
    ] + list(RESCORING_FUNCTIONS.keys()),
    help="The method(s) to use for pose clustering. Must be one or more of:\n" +
    "- RMSD : Cluster compounds on RMSD matrix of poses \n" +
    "- spyRMSD : Cluster compounds on symmetry-corrected RMSD matrix of poses\n"
    +
    "- espsim : Cluster compounds on electrostatic shape similarity matrix of poses\n"
    + "- USRCAT : Cluster compounds on shape similarity matrix of poses\n" +
    "- 3DScore : Selects pose with the lowest average RMSD to all other poses\n"
    + "- bestpose : Takes the best pose from each docking program\n" +
    "- bestpose_GNINA : Takes the best pose from GNINA docking program\n" +
    "- bestpose_SMINA : Takes the best pose from SMINA docking program\n" +
    "- bestpose_QVINAW : Takes the best pose from QVINAW docking program\n" +
    "- bestpose_QVINA2 : Takes the best pose from QVINA2 docking program\n" +
    "- bestpose_PLANTS : Takes the best pose from PLANTS docking program  \n" +
    "- You can also use any of the scoring functions and DockM8 will select the best pose for each compound according to the specified scoring function.",
)
# Clustering algorithm
if any(x in CLUSTERING_METRICS.keys() for x in pose_selection):
    clustering_algorithm = st.selectbox(
        label="Which clustering algorithm do you want to use?",
        options=("KMedoids", "Aff_Prop"),
        index=0,
        help=
        'Which algorithm to use for clustering. Must be one of "KMedoids", "Aff_prop". Must be set when using "RMSD", "spyRMSD", "espsim", "USRCAT" clustering metrics.',
    )
else:
    clustering_algorithm = None

# Rescoring
st.header("Scoring functions", divider="orange")
rescoring = st.multiselect(
    label="Choose the scoring functions you want to use",
    default=["CNN-Score", "KORP-PL"],
    options=list(RESCORING_FUNCTIONS.keys()),
    help="The method(s) to use for scoring. Multiple selection allowed",
)

# Consensus
st.header("Consensus", divider="orange")
consensus_method = st.selectbox(
    label="Choose which consensus algorithm to use: ",
    index=2,
    options=list(CONSENSUS_METHODS.keys()),
    help="The method to use for consensus.",
)

if gen_decoys:
    total_combinations = 0
    for length in range(2, len(rescoring)):
        combinations = list(itertools.combinations(rescoring, length))
        total_combinations += len(combinations)
    num_possibilities = (len(CONSENSUS_METHODS.keys()) * len(pose_selection) *
                         (len(rescoring) + total_combinations))
    if num_possibilities > 10000:
        st.warning(
            f"WARNING: The combination of scoring functions and pose selection method you have selected will yield a large number of possible combinations ({num_possibilities}). This may take a long time to run."
        )

config = {
    "general": {
        "software": software,
        "mode": mode.lower(),
        "n_cpus": n_cpus
    },
    "decoy_generation": {
        "gen_decoys": gen_decoys,
        "decoy_model": decoy_model,
        "n_decoys": n_decoys,
        "actives": actives,
    },
    "receptor(s)": [str(receptor) for receptor in receptors],
    "docking_library": docking_library,
    "protein_preparation": {
        "select_best_chain": select_best_chain,
        "fix_nonstandard_residues": fix_nonstandard_residues,
        "fix_missing_residues": fix_missing_residues,
        "remove_heteroatoms": remove_heteroatoms,
        "remove_water": remove_water,
        "add_hydrogens": add_hydrogens,
        "protonation": protonation,
    },
    "ligand_preparation": {
        "protonation": ligand_protonation,
        "conformers": ligand_conformers,
        "n_conformers": n_conformers,
    },
    "pocket_detection": {
        "method":
            pocket_mode,
        "reference_ligand(s)": [str(ligand) for ligand in reference_files]
                               if reference_files is not None else None,
        "radius":
            pocket_radius,
        "manual_pocket":
            manual_pocket,
    },
    "docking": {
        "docking_programs": docking_programs,
        "n_poses": n_poses,
        "exhaustiveness": exhaustiveness,
    },
    "post_docking": {
        "clash_cutoff": clash_cutoff,
        "strain_cutoff": strain_cutoff,
        "bust_poses": bust_poses,
    },
    "pose_selection": {
        "pose_selection_method": pose_selection,
        "clustering_method": clustering_algorithm,
    },
    "rescoring": rescoring,
    "consensus": consensus_method,
    "threshold": threshold,
}

open("log.txt", "w").close()

import datetime
import yaml


def read_log_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content


# Run the script file
if st.button("Run DockM8"):
    # Generate the file name with date and time
    now = datetime.datetime.now()
    file_name = now.strftime("%Y-%m-%d_%H-%M") + "_dockm8_config.yml"

    # Write the config to the YAML file
    with open(file_name, "w") as f:
        yaml.dump(config, f)

    # Print the file path
    st.write(f"Config file saved as: {file_name}. Starting DockM8 ...")
    command = f"{sys.executable} {CWD}/dockm8.py --config {CWD + '/' + file_name}"
    subprocess.Popen(command, shell=True)

log_file_path = Path(CWD, "log.txt")

if log_file_path is not None:
    log_content = read_log_file(log_file_path)
    # Create an empty container for dynamic content updates
    log_container = st.empty()
    # Display initial log content
    log_container.text_area("Log ", log_content, height=300)
    # Periodically check for changes in the log file
    while True:
        time.sleep(0.2)  # Adjust the interval as needed
        new_log_content = read_log_file(log_file_path)
        if new_log_content != log_content:
            # Update the contents of the existing text area
            log_container.text_area("Log ", new_log_content, height=300)
            log_content = new_log_content
