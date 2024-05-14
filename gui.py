# Import necessary libraries
import itertools
import os
import shlex
import subprocess
import sys
import time
import warnings
from pathlib import Path
import streamlit as st

from scripts.clustering_metrics import CLUSTERING_METRICS
from scripts.consensus_methods import CONSENSUS_METHODS
from scripts.docking_functions import DOCKING_PROGRAMS
from scripts.rescoring_functions import RESCORING_FUNCTIONS

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(page_title="DockM8 v1.0.0", page_icon="./media/DockM8_logo.png", layout="wide")
# Sidebar
st.sidebar.image(image="./media/DockM8_white_horizontal.png", width=200)
st.sidebar.title("DockM8")
st.sidebar.subheader("Open-source consensus docking for everyone")
st.sidebar.link_button("Github", url="https://github.com/DrugBud-Suite/DockM8")
st.sidebar.link_button("Visit Website", url="https://drugbud-suite.github.io/dockm8-web/")
st.sidebar.link_button("Publication", url="https://doi.org/your-doi")
st.sidebar.link_button("Zenodo repository", url="https://doi.org/your-doi")

# Logo
st.columns(3)[1].image(image="./media/DockM8_white_vertical.png", width=400)

# Setup
CWD = os.getcwd()
st.header("Setup", divider="orange")
mode = st.selectbox(
    label="Which mode do you want to run DockM8 in?",
    key=0,
    options=("Single", "Ensemble"),
    help="Single Docking: DockM8 will dock each ligand in the library to a single receptor. "
    + "Ensemble Docking: DockM8 will dock each ligand in the library to all specified receptors and then combine the results to create an ensemble consensus.",
)
if mode != "Single":
    threshold = st.slider(
        label="Threshold for ensemble consensus (in %)",
        min_value=0.0,
        max_value=10.0,
        step=0.1,
        value=1.0,
        help="Threshold for ensemble consensus (in %). DockM8 will only consider a ligand as a consensus hit if it is a top scorer for all the receptors.",
    )
num_cpus = st.slider(
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
    help="Type the directory containing the software folder: For example: /home/user/Dockm8/software",
)

gen_decoys = st.toggle(
    label="Generate decoys",
    value=False,
    help="Generate decoys for the active ligands and determine optimal DockM8 conditions",
)

if mode == "Single":
    receptor_value = CWD + "/dockm8_testing/4kd1_p.pdb"
    reference_value = CWD + "/dockm8_testing/4kd1_l.sdf"
    library_value = CWD + "/dockm8_testing/library.sdf"
if mode == "Ensemble":
    receptor_value = CWD + "/dockm8_testing/4kd1_p.pdb, " + CWD + "/dockm8_testing/1fvv_p.pdb"
    reference_value = CWD + "/dockm8_testing/4kd1_l.sdf, " + CWD + "/dockm8_testing/1fvv_l.sdf"
    library_value = CWD + "/dockm8_testing/library.sdf"

if gen_decoys:
    st.subheader("Decoy generation", divider="orange")
    # Active ligands
    active_ligands = st.text_input(
        label="Enter the path to the active ligands file (.sdf format)",
        value = CWD + "/dockm8_testing/CDK2_actives.sdf",
        help="Choose an active ligands file (.sdf format)",
    )
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

col1, col2 = st.columns(2)

# Receptor(s)
col1.header("Receptor(s)", divider="orange")
receptor_file = col1.text_input(
    label="File path(s) of one or more multiple receptor files (.pdb format), separated by commas",
    help="Choose one or multiple receptor files (.pdb format)",
    value=receptor_value,
    placeholder="Enter path(s) here",
)

# Prepare receptor
prepare_receptor = col1.toggle(
    label="Prepare receptor using Protoss",
    value=True,
    help="Choose whether or not to use Protoss Web service to protonate the protein structure",
)

# Pocket finding
col1.subheader("Pocket finding", divider="orange")
pocket_mode = col1.selectbox(
    label="How should the pocket be defined?",
    options=("Reference", "RoG", "Dogsitescorer", "Custom"),
    help="Reference Ligand: DockM8 will use the reference ligand to define the pocket. "
    + "Reference Ligand RoG: DockM8 will use the reference ligand radius of gyration. "
    + "DogSiteScorer: DockM8 will use the DogSiteScorer pocket finding algorithm to define the pocket."
    + "Custom: Define your own pocket center and size coordinates."
)

# Reference ligand
if pocket_mode == "Reference" or pocket_mode == "RoG":
    reference_file = col1.text_input(
        label="File path(s) of one or more multiple reference ligand files (.sdf format), separated by commas",
        help="Choose one or multiple reference ligand files (.pdb format)",
        value=reference_value,
        placeholder="Enter path(s) here",
    )
elif pocket_mode == "Custom" and mode == "Single":
    ccol1, ccol2, ccol3 = col1.columns(3)
    x_center = ccol1.number_input(label="X Center", value=0.0, help="Enter the X coordinate of the pocket center")
    y_center = ccol2.number_input(label="Y Center", value=0.0, help="Enter the Y coordinate of the pocket center")
    z_center = ccol3.number_input(label="Z Center", value=0.0, help="Enter the Z coordinate of the pocket center")
    x_size = ccol1.number_input(label="X Size", value=20.0, help="Enter the size of the pocket in the X direction")
    y_size = ccol2.number_input(label="Y Size", value=20.0, help="Enter the size of the pocket in the Y direction")
    z_size = ccol3.number_input(label="Z Size", value=20.0, help="Enter the size of the pocket in the Z direction")
    pocket_coordinates = {"center": [x_center,y_center,z_center],
                          "size": [x_size,y_size,z_size]}
elif pocket_mode == "Custom" and mode != "Single":
    col1.error("Custom pocket definition does not currently work in ensemble mode, please change the pocket definition mode")

# Ligand library
col2.header("Ligands", divider="orange")
ligand_file = col2.text_input(
    label="Entre the path to the ligand library file (.sdf format)",
    value=library_value,
    help="Choose a ligand library file (.sdf format)",
)

# ID column
id_column = col2.text_input(
    label="Choose the column name that contains the ID of the ligand",
    value="ID",
    help="Choose the column name that contains the ID of the ligand",
)

# Ligand conformers
col2.subheader("Ligand conformers", divider="orange")
ligand_conformers = col2.selectbox(
    label="How should the conformers be generated?",
    options=["MMFF", "GypsumDL"],
    index=1,
    help="MMFF: DockM8 will use MMFF to prepare the ligand 3D conformers. "
    + "GypsumDL: DockM8 will use Gypsum-DL to prepare the ligand 3D conformers.",
)

# Ligand protonation
col2.subheader("Ligand protonation", divider="orange")
ligand_protonation = col2.selectbox(
    label="How should the ligands be protonated?",
    options=("None", "GypsumDL"),
    index=1,
    help="None: No protonation "
    + "Gypsum-DL: DockM8 will use  Gypsum-DL to protonate the ligands",
)

# Docking programs
st.header("Docking programs", divider="orange")
docking_programs = st.multiselect(
    label="Choose the docking programs you want to use",
    default=["GNINA"],
    options=DOCKING_PROGRAMS,
    help="Choose the docking programs you want to use, mutliple selection is allowed",
)

if "PLANTS" in docking_programs and not os.path.exists("/path/to/software/PLANTS"):
    st.warning('PLANTS was not found in the software folder, please visit http://www.tcd.uni-konstanz.de/research/plants.php', icon=':warning:')

# Number of poses
nposes = st.slider(
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
    help="Exhaustiveness of the docking, only applies to GNINA, SMINA, QVINA2 and QVINAW. Higher values can significantly increase the runtime.",
)

bust_poses = st.checkbox(
    label="Bust poses using PoseBusters : WARNING may take a long time to run",
    value=False,
    help="Bust poses using PoseBusters : Will remove any poses with clashes, non-flat aromatic rings etc. WARNING may take a long time to run",
)

# Pose selection
st.header("Pose Selection", divider="orange")
pose_selection = st.multiselect(
    label="Choose the pose selection method you want to use",
    default=["KORP-PL"],
    options=list(CLUSTERING_METRICS.keys())
    + [
        "bestpose",
        "bestpose_GNINA",
        "bestpose_SMINA",
        "bestpose_PLANTS",
        "bestpose_QVINA2",
        "bestpose_QVINAW",
    ]
    + list(RESCORING_FUNCTIONS.keys()),
    help="The method(s) to use for pose clustering. Must be one or more of:\n"
    + "- RMSD : Cluster compounds on RMSD matrix of poses \n"
    + "- spyRMSD : Cluster compounds on symmetry-corrected RMSD matrix of poses\n"
    + "- espsim : Cluster compounds on electrostatic shape similarity matrix of poses\n"
    + "- USRCAT : Cluster compounds on shape similarity matrix of poses\n"
    + "- 3DScore : Selects pose with the lowest average RMSD to all other poses\n"
    + "- bestpose : Takes the best pose from each docking program\n"
    + "- bestpose_GNINA : Takes the best pose from GNINA docking program\n"
    + "- bestpose_SMINA : Takes the best pose from SMINA docking program\n"
    + "- bestpose_QVINAW : Takes the best pose from QVINAW docking program\n"
    + "- bestpose_QVINA2 : Takes the best pose from QVINA2 docking program\n"
    + "- bestpose_PLANTS : Takes the best pose from PLANTS docking program  \n"
    + "- You can also use any of the scoring functions and DockM8 will select the best pose for each compound according to the specified scoring function.",
)
# Clustering algorithm
if any(x in CLUSTERING_METRICS.keys() for x in pose_selection):
    clustering_algorithm = st.selectbox(
        label="Which clustering algorithm do you want to use?",
        options=("KMedoids", "Aff_Prop"),
        index=0,
        help='Which algorithm to use for clustering. Must be one of "KMedoids", "Aff_prop". Must be set when using "RMSD", "spyRMSD", "espsim", "USRCAT" clustering metrics.',
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
    num_possibilities = (
        len(CONSENSUS_METHODS.keys())
        * len(pose_selection)
        * (len(rescoring) + total_combinations)
    )
    if num_possibilities > 10000:
        st.warning(
            f"WARNING: The combination of scoring functions and pose selection method you have selected will yield a large number of possible combinations ({num_possibilities}). This may take a long time to run."
        )

command = (
    f'{sys.executable} {CWD}/dockm8.py '
    f'--software {software} '
    f'--receptor {receptor_file} '
    f'--docking_library {ligand_file} '
    f'--idcolumn {id_column} '
    f'--prepare_proteins {prepare_receptor} '
    f'--conformers {ligand_conformers} '
    f'--protonation {ligand_protonation} '
    f'--docking_programs {" ".join(docking_programs)} '
    f'--bust_poses {bust_poses} '
    f'--pose_selection {" ".join(pose_selection)} '
    f'--nposes {nposes} '
    f'--exhaustiveness {exhaustiveness} '
    f'--ncpus {num_cpus} '
    f'--clustering_method {clustering_algorithm} '
    f'--rescoring {" ".join(rescoring)} '
    f'--consensus {consensus_method}'
)
# Add pocket-specific arguments
if pocket_mode == "Custom":
    pocket_str = '*'.join([f"{k}:{','.join(map(str, v))}" for k, v in pocket_coordinates.items()])
    command += (f" --pocket {pocket_str}")
elif pocket_mode == "Reference" or pocket_mode == "RoG":
    command += (f" --pocket {pocket_mode}")
    command += (f" --reffile {reference_file}")
elif pocket_mode == "Dogsitescorer":
    command += (f" --pocket {pocket_mode}")

# Add mode-specific arguments
if mode == "ensemble" or mode == "active_learning":
    command += f" --mode {mode} --threshold {threshold}"
else:
    command += f" --mode {mode}"

if gen_decoys:
    command += (
        " --gen_decoys True "
        f"--decoy_model {decoy_model} "
        f"--n_decoys {n_decoys} "
        f"--actives {active_ligands} "
    )

open("log.txt", "w").close()


def run_dockm8(command_list):
    print("Running")
    subprocess.Popen(command_list)


def read_log_file(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content


# Run the script file
if st.button("Run DockM8"):
    command_list = shlex.split(command)
    run_dockm8(command_list)

log_file_path = Path(CWD, "log.txt")

if log_file_path is not None:
    log_content = read_log_file(log_file_path)
    # Create an empty container for dynamic content updates
    log_container = st.empty()
    # Display initial log content
    log_container.text_area("Log ", log_content, height=300)
    # Periodically check for changes in the log file
    while True:
        time.sleep(1)  # Adjust the interval as needed
        new_log_content = read_log_file(log_file_path)
        if new_log_content != log_content:
            # Update the contents of the existing text area
            log_container.text_area("Log ", new_log_content, height=300)
            log_content = new_log_content
