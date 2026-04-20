# Import necessary libraries
import itertools
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
import streamlit as st


from scripts.consensus.consensus import _METHODS
from scripts.docking.docking import DOCKING_PROGRAMS
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(page_title="DockM8 v1.1.0", page_icon="./media/dockm8_white_logoonly.png", layout="wide")

# Utility functions
def bool_to_str(value):
    """Convert Python boolean to lowercase string for command line args"""
    return str(value).lower()

def validate_file_exists(path, file_type="File"):
    """Check if a file exists and return an error message if not."""
    if not path or not path.strip():
        return f"{file_type} path is empty"

    file_path = Path(path.strip())
    if not file_path.exists():
        return f"{file_type} not found: {path}"
    return None

def validate_dir_exists(path, dir_type="Directory"):
    """Check if a directory exists and return an error message if not."""
    if not path or not path.strip():
        return f"{dir_type} path is empty"

    dir_path = Path(path.strip())
    if not dir_path.exists():
        return f"{dir_type} not found: {path}"
    if not dir_path.is_dir():
        return f"{path} is not a directory"
    return None

# Initialize variables that may be conditionally defined
active_ligands = ""
active_ligands_error = None
plants_error = None
custom_pocket_ensemble_error = None

# Selection method options: bestpose + all rescoring functions
SELECTION_METHOD_OPTIONS = ["bestpose"] + list(RESCORING_FUNCTIONS.keys())
DOCKING_PROGRAM_OPTIONS = list(DOCKING_PROGRAMS.keys())

# Sidebar
st.sidebar.image(image="./media/dockm8_white.png", width=200)
st.sidebar.title("V1.1.0")
st.sidebar.subheader("Open-source consensus docking for everyone", divider="orange")
st.sidebar.link_button("Github", url="https://github.com/DrugBud-Suite/DockM8", use_container_width=True)
st.sidebar.link_button("Visit Website", url="https://drugbud-suite.github.io/dockm8-web/", use_container_width=True)
st.sidebar.link_button("Publication", url="https://doi.org/TODO-REPLACE-WITH-DOI", use_container_width=True)

with st.sidebar.expander("Benchmark datasets (Zenodo)"):
    st.markdown(
        "- **DEKOIS 2.0** — [10.5281/zenodo.15430058](https://doi.org/10.5281/zenodo.15430058)\n"
        "- **DUD-E** — [10.5281/zenodo.15430186](https://doi.org/10.5281/zenodo.15430186)\n"
        "- **Lit-PCBA (Part 1/3)** — [10.5281/zenodo.16436211](https://doi.org/10.5281/zenodo.16436211)\n"
        "- **Lit-PCBA (Part 2/3)** — [10.5281/zenodo.16436304](https://doi.org/10.5281/zenodo.16436304)\n"
        "- **Lit-PCBA (Part 3/3)** — [10.5281/zenodo.16436306](https://doi.org/10.5281/zenodo.16436306)"
    )

# Logo
st.columns(3)[1].image(image="./media/dockm8_white.png", width=400)

# Setup
CWD = os.getcwd()
st.header("Setup", divider="orange")
mode = st.selectbox(
    label="Which mode do you want to run DockM8 in?",
    key=0,
    options=("single", "ensemble"),
    help="Single Docking: DockM8 will dock each ligand in the library to a single receptor. "
    + "Ensemble Docking: DockM8 will dock each ligand in the library to all specified receptors and then combine the results to create an ensemble consensus.",
)
if mode == "ensemble":
    ensemble_top_percent = st.slider(
        label="Top percentage to consider (in %)",
        min_value=0.1,
        max_value=20.0,
        step=0.1,
        value=1.0,
        help="Report compounds in the top X percent across all structures when using ensemble mode."
    )
num_cpus = st.slider(
    "Number of CPUs",
    min_value=1,
    max_value=os.cpu_count(),
    step=1,
    value=int(os.cpu_count() * 0.9),
    help="Number of CPUs to use for calculations",
)
software_path = st.text_input(
    "Choose a software directory",
    value=CWD + "/software",
    help="Type the directory containing the software folder: For example: /home/user/Dockm8/software",
)

# Validate software directory
software_error = validate_dir_exists(software_path, "Software directory")
if software_error:
    st.error(software_error)

gen_decoys = st.toggle(
    label="Generate decoys",
    value=False,
    help="Generate decoys for the active ligands and determine optimal DockM8 conditions",
)

# Validation: decoy generation + ensemble = not allowed
if gen_decoys and mode == "ensemble":
    st.error("Decoy generation is not supported in ensemble mode. Please use single mode or disable decoy generation.")

# Initialize pocket method in session state
if 'pocket_mode_display' not in st.session_state:
    st.session_state.pocket_mode_display = "Reference"

# Initialize session state for receptor paths if not exists
if 'receptor_paths' not in st.session_state:
    if mode == "single":
        st.session_state.receptor_paths = [CWD + "/test_data/4kd1_p.pdb"]
    else:
        st.session_state.receptor_paths = [
            CWD + "/test_data/4kd1_p.pdb",
            CWD + "/test_data/1fvv_p.pdb"
        ]

# Initialize session state for reference ligand paths if not exists
if 'reference_ligand_paths' not in st.session_state:
    if mode == "single":
        st.session_state.reference_ligand_paths = [CWD + "/test_data/4kd1_l.sdf"]
    else:
        st.session_state.reference_ligand_paths = [
            CWD + "/test_data/4kd1_l.sdf",
            CWD + "/test_data/1fvv_l.sdf"
        ]

if gen_decoys:
    st.subheader("Decoy generation", divider="orange")
    # Active ligands
    active_ligands = st.text_input(
        label="Enter the path to the active ligands file (.sdf format)",
        value=CWD + "/test_data/CDK2_actives.sdf",
        help="Choose an active ligands file (.sdf format)",
    )

    # Validate active ligands file
    active_ligands_error = validate_file_exists(active_ligands, "Active ligands file")
    if active_ligands_error:
        st.error(active_ligands_error)

    # Number of decoys
    n_decoys = st.slider(
        label="Number of decoys",
        min_value=1,
        max_value=100,
        step=1,
        value=20,
        help="Number of decoys to generate for each active ligand",
    )
    # Decoy generation program
    decoy_model_options = {
        "DUD-E": "DUDE",
        "DEKOIS": "DEKOIS",
        "DUD-E_phosphorus": "DUDE_P"
    }
    decoy_model_display = st.selectbox(
        label="Which decoy generation model do you want to use?",
        options=list(decoy_model_options.keys()),
        help="Select which Deepcoy decoy generation model you want to use ",
    )
    decoy_model = decoy_model_options[decoy_model_display]

col1, col2 = st.columns(2)

# Receptor(s)
col1.header("Receptor(s)", divider="orange")

# Functions for managing receptors and reference ligands
def add_receptor():
    st.session_state.receptor_paths.append("")
    # Add a corresponding reference ligand path if we're using Reference or RoG
    if st.session_state.pocket_mode_display in ["Reference", "RoG"]:
        st.session_state.reference_ligand_paths.append("")

def remove_receptor(index):
    if len(st.session_state.receptor_paths) > 1:
        st.session_state.receptor_paths.pop(index)
        if len(st.session_state.reference_ligand_paths) > index:
            st.session_state.reference_ligand_paths.pop(index)

# Display receptors with validation
receptor_errors = []
for i, path in enumerate(st.session_state.receptor_paths):
    cols = col1.columns([0.8, 0.2])
    new_path = cols[0].text_input(
        f"Receptor {i+1}",
        value=path,
        key=f"receptor_{i}"
    )
    st.session_state.receptor_paths[i] = new_path

    # Validate receptor file
    error = validate_file_exists(new_path, f"Receptor {i+1} file")
    if error:
        receptor_errors.append(error)
        cols[0].error(error)

    if len(st.session_state.receptor_paths) > 1:
        if cols[1].button("Remove", key=f"remove_receptor_{i}"):
            remove_receptor(i)

# Add receptor button
if col1.button("Add Receptor"):
    add_receptor()

# Validate receptors for ensemble mode
if mode == "ensemble" and len([p for p in st.session_state.receptor_paths if p.strip()]) < 2:
    ensemble_error = "Ensemble mode requires at least two valid receptors"
    col1.error(ensemble_error)
    receptor_errors.append(ensemble_error)

# Prepare receptor
prepare_receptor = col1.toggle(
    label="Prepare receptor",
    value=True,
    help="Choose whether or not to prepare the protein structure (protonation, residue fixes, etc.)",
)

# Advanced Protein Preparation options - only shown when prepare_receptor is True
if prepare_receptor:
    with col1.expander("Advanced Protein Preparation"):
        protein_protonation = st.selectbox(
            label="Protonation method",
            options=["protoss", "pdbfixer", "none"],
            index=0,
            help="Method for protein protonation. Protoss: Use Protoss Web service. PDBFixer: Use PDBFixer. None: Skip protonation.",
        )
        fix_nonstandard_residues = st.toggle(
            label="Fix nonstandard residues",
            value=True,
            help="Fix nonstandard residues during protein preparation",
        )
        fix_missing_residues = st.toggle(
            label="Fix missing residues",
            value=True,
            help="Fix missing residues during protein preparation",
        )
        remove_hetero = st.toggle(
            label="Remove heteroatoms",
            value=True,
            help="Remove heteroatoms during protein preparation",
        )
        remove_water = st.toggle(
            label="Remove water molecules",
            value=True,
            help="Remove water molecules during protein preparation",
        )
else:
    protein_protonation = "protoss"
    fix_nonstandard_residues = True
    fix_missing_residues = True
    remove_hetero = True
    remove_water = True

# Pocket finding
col1.subheader("Pocket finding", divider="orange")
pocket_mode_options = {
    "Reference": "Reference",
    "RoG": "RoG",
    "Dogsitescorer": "Dogsitescorer",
    "Custom": "Manual"
}
pocket_mode_display = col1.selectbox(
    label="How should the pocket be defined?",
    options=list(pocket_mode_options.keys()),
    help="Reference Ligand: DockM8 will use the reference ligand to define the pocket. "
    + "Reference Ligand RoG: DockM8 will use the reference ligand radius of gyration. "
    + "DogSiteScorer: DockM8 will use the DogSiteScorer pocket finding algorithm to define the pocket. "
    + "Custom: Define your own pocket center and size coordinates.",
)
# Update the pocket mode in session state
st.session_state.pocket_mode_display = pocket_mode_display

pocket_mode = pocket_mode_options[pocket_mode_display]

# Check if Custom pocket is selected in ensemble mode
if pocket_mode_display == "Custom" and mode == "ensemble":
    custom_pocket_ensemble_error = "Custom pocket definition does not currently work in ensemble mode, please change the pocket definition mode"
    col1.error(custom_pocket_ensemble_error)

# Pocket radius (applies to Reference and RoG modes)
pocket_radius = 10
if pocket_mode_display in ["Reference", "RoG"]:
    pocket_radius = col1.number_input(
        label="Pocket radius",
        value=10,
        min_value=1,
        step=1,
        help="Radius for pocket finding (in Angstroms)",
    )

# Reference ligand with validation
reference_ligand_errors = []
if pocket_mode_display in ["Reference", "RoG"]:
    col1.subheader("Reference Ligands", divider="orange")

    # Ensure reference_ligand_paths has same length as receptor_paths
    while len(st.session_state.reference_ligand_paths) < len(st.session_state.receptor_paths):
        st.session_state.reference_ligand_paths.append("")

    for i, path in enumerate(st.session_state.reference_ligand_paths[:len(st.session_state.receptor_paths)]):
        ref_path = col1.text_input(
            f"Reference Ligand {i+1} (for Receptor {i+1})",
            value=path,
            key=f"ref_ligand_{i}"
        )
        st.session_state.reference_ligand_paths[i] = ref_path

        # Validate reference ligand file
        error = validate_file_exists(ref_path, f"Reference ligand {i+1} file")
        if error and st.session_state.receptor_paths[i].strip():
            reference_ligand_errors.append(error)
            col1.error(error)

    # Validate reference ligand count matches receptor count in ensemble mode
    if mode == "ensemble":
        valid_receptors = [p for p in st.session_state.receptor_paths if p.strip()]
        valid_refs = [ref for i, ref in enumerate(st.session_state.reference_ligand_paths)
                     if i < len(st.session_state.receptor_paths) and ref.strip() and st.session_state.receptor_paths[i].strip()]
        if len(valid_refs) != len(valid_receptors) and len(valid_receptors) > 0:
            ref_count_error = f"Number of reference ligands ({len(valid_refs)}) must match number of receptors ({len(valid_receptors)}) in ensemble mode"
            col1.error(ref_count_error)
            reference_ligand_errors.append(ref_count_error)

elif pocket_mode_display == "Custom" and mode == "single":
    ccol1, ccol2, ccol3 = col1.columns(3)
    x_center = ccol1.number_input(label="X Center", value=0.0, help="Enter the X coordinate of the pocket center")
    y_center = ccol2.number_input(label="Y Center", value=0.0, help="Enter the Y coordinate of the pocket center")
    z_center = ccol3.number_input(label="Z Center", value=0.0, help="Enter the Z coordinate of the pocket center")
    x_size = ccol1.number_input(label="X Size", value=20.0, help="Enter the size of the pocket in the X direction")
    y_size = ccol2.number_input(label="Y Size", value=20.0, help="Enter the size of the pocket in the Y direction")
    z_size = ccol3.number_input(label="Z Size", value=20.0, help="Enter the size of the pocket in the Z direction")
    pocket_coordinates = {"center": [x_center, y_center, z_center], "size": [x_size, y_size, z_size]}

if pocket_mode_display == "Dogsitescorer":
    dss_mode = col1.selectbox(
        label="Choose method to select binding site",
        options=("Volume", "Druggability_Score", "Surface", "Depth"),
        help="Choose which DogSiteScorer metric to use to select the binding site. The site with the highest value of the chosen metric will be used.",
    )

# Ligand library
col2.header("Ligands", divider="orange")
ligand_file = col2.text_input(
    label="Enter the path to the ligand library file (.sdf format)",
    value=CWD + "/test_data/library.sdf",
    help="Choose a ligand library file (.sdf format)",
)

# Validate ligand file
ligand_file_error = validate_file_exists(ligand_file, "Ligand library file")
if ligand_file_error:
    col2.error(ligand_file_error)

# ID column
id_column = col2.text_input(
    label="Choose the column name that contains the ID of the ligand",
    value="ID",
    help="Choose the column name that contains the ID of the ligand",
)

# Ligand conformers
col2.subheader("Ligand conformers", divider="orange")
conformer_options = ["UFF", "MMFF", "GypsumDL"]
ligand_conformers = col2.selectbox(
    label="How should the conformers be generated?",
    options=conformer_options,
    index=conformer_options.index("MMFF"),
    help="UFF: Use UFF force field for conformer generation. "
    + "MMFF: Use MMFF force field for conformer generation. "
    + "GypsumDL: Use Gypsum-DL for conformer generation.",
)

# Ligand protonation
col2.subheader("Ligand protonation", divider="orange")
ligand_protonation = col2.selectbox(
    label="How should the ligands be protonated?",
    options=("None", "GypsumDL"),
    index=1,
    help="None: No protonation. Gypsum-DL: DockM8 will use Gypsum-DL to protonate the ligands.",
)

# Advanced Ligand Preparation options
with col2.expander("Advanced Ligand Preparation"):
    standardize_ids = st.toggle(
        label="Standardize compound IDs",
        value=True,
        help="Standardize compound IDs during ligand preparation",
    )
    remove_salts = st.toggle(
        label="Remove salts",
        value=True,
        help="Remove salts during ligand preparation",
    )
    min_ph = st.number_input(
        label="Minimum pH",
        value=6.4,
        step=0.1,
        help="Minimum pH for ligand protonation (GypsumDL)",
    )
    max_ph = st.number_input(
        label="Maximum pH",
        value=8.4,
        step=0.1,
        help="Maximum pH for ligand protonation (GypsumDL)",
    )
    pka_precision = st.number_input(
        label="pKa precision",
        value=1.0,
        step=0.1,
        help="pKa precision for ligand protonation (GypsumDL)",
    )

# Docking programs
st.header("Docking programs", divider="orange")

# If not in decoy generation mode, limit to one docking program
if not gen_decoys:
    st.info("In standard mode, only one docking program and one pose selection method can be used. Enable decoy generation to test multiple combinations.")
    docking_program = st.selectbox(
        label="Choose the docking program you want to use",
        options=DOCKING_PROGRAM_OPTIONS,
        index=DOCKING_PROGRAM_OPTIONS.index("GNINA") if "GNINA" in DOCKING_PROGRAM_OPTIONS else 0,
        help="Choose the docking program you want to use"
    )
    docking_programs = [docking_program]
else:
    docking_programs = st.multiselect(
        label="Choose the docking programs you want to use",
        default=["GNINA"],
        options=DOCKING_PROGRAM_OPTIONS,
        help="Choose the docking programs you want to use, multiple selection is allowed in decoy generation mode"
    )

# Check if PLANTS is selected but software path doesn't contain it
if "PLANTS" in docking_programs:
    plants_path = Path(software_path) / "PLANTS"
    if not plants_path.exists():
        plants_error = "PLANTS was not found in the software folder, please visit http://www.tcd.uni-konstanz.de/research/plants.php"
        st.warning(plants_error, icon=":warning:")

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

# Pose selection - Handle single vs multiple selection based on decoy mode
st.header("Pose Selection", divider="orange")

if not gen_decoys:
    selection_method = st.selectbox(
        label="Choose the pose selection method you want to use",
        options=SELECTION_METHOD_OPTIONS,
        index=SELECTION_METHOD_OPTIONS.index("bestpose"),
        help="Select the method to use for pose selection. 'bestpose' selects the best-scored pose from the docking program itself (fastest option)."
    )
    selection_methods = [selection_method]
else:
    selection_methods = st.multiselect(
        label="Choose the pose selection methods you want to use",
        default=["bestpose"],
        options=SELECTION_METHOD_OPTIONS,
        help="You can use 'bestpose' or any of the scoring functions. DockM8 will select the best pose for each compound according to the specified method."
    )


# Rescoring
st.header("Scoring functions", divider="orange")
rescoring_functions = st.multiselect(
    label="Choose the scoring functions you want to use",
    default=["GNINA-Affinity", "CNN-Score"],
    options=list(RESCORING_FUNCTIONS.keys()),
    help="The method(s) to use for scoring. Multiple selection allowed",
)

# Consensus
st.header("Consensus", divider="orange")
consensus_methods_list = list(_METHODS.keys())
consensus_method = st.selectbox(
    label="Choose which consensus algorithm to use: ",
    options=consensus_methods_list,
    index=consensus_methods_list.index("rbr") if "rbr" in consensus_methods_list else 0,
    help="The method to use for consensus.",
)

if gen_decoys:
    total_combinations = 0
    for length in range(2, len(rescoring_functions)):
        combinations = list(itertools.combinations(rescoring_functions, length))
        total_combinations += len(combinations)
    num_possibilities = len(_METHODS.keys()) * len(selection_methods) * (len(rescoring_functions) + total_combinations)
    if num_possibilities > 10000:
        st.warning(
            f"WARNING: The combination of scoring functions and pose selection method you have selected will yield a large number of possible combinations ({num_possibilities}). This may take a long time to run."
        )

# Construct command as a list for robust path handling
command_list = [sys.executable, f"{CWD}/dockm8.py"]

# Required arguments - always included
command_list.extend(["--software", software_path])

# Filter valid receptor paths
valid_receptor_paths = [p for p in st.session_state.receptor_paths if p.strip()]
if valid_receptor_paths:
    command_list.append("--receptor")
    command_list.extend(valid_receptor_paths)

command_list.extend(["--mode", mode])
command_list.extend(["--docking_library", ligand_file])
command_list.extend(["--id_column", id_column])
command_list.extend(["--ligand_protonation", ligand_protonation])

# Protein preparation
command_list.extend(["--prepare_protein", bool_to_str(prepare_receptor)])
if prepare_receptor:
    if protein_protonation != "protoss":
        command_list.extend(["--protein_protonation", protein_protonation])
    if not fix_nonstandard_residues:
        command_list.extend(["--fix_nonstandard_residues", "false"])
    if not fix_missing_residues:
        command_list.extend(["--fix_missing_residues", "false"])
    if not remove_hetero:
        command_list.extend(["--remove_hetero", "false"])
    if not remove_water:
        command_list.extend(["--remove_water", "false"])

# Ligand preparation - only pass non-defaults
if ligand_conformers != "MMFF":
    command_list.extend(["--conformers", ligand_conformers])
if not standardize_ids:
    command_list.extend(["--standardize_ids", "false"])
if not remove_salts:
    command_list.extend(["--remove_salts", "false"])
if min_ph != 6.4:
    command_list.extend(["--min_ph", str(min_ph)])
if max_ph != 8.4:
    command_list.extend(["--max_ph", str(max_ph)])
if pka_precision != 1.0:
    command_list.extend(["--pka_precision", str(pka_precision)])

# Docking programs
if docking_programs:
    command_list.append("--docking_program")
    command_list.extend(docking_programs)

# Docking options - only pass non-defaults
command_list.extend(["--bust_poses", bool_to_str(bust_poses)])
if nposes != 10:
    command_list.extend(["--n_poses", str(nposes)])
if exhaustiveness != 8:
    command_list.extend(["--exhaustiveness", str(exhaustiveness)])
if num_cpus != int(os.cpu_count() * 0.9):
    command_list.extend(["--n_cpus", str(num_cpus)])

# Selection methods
if selection_methods:
    command_list.append("--selection_method")
    command_list.extend(selection_methods)

# Rescoring functions
if rescoring_functions:
    command_list.append("--rescoring_functions")
    command_list.extend(rescoring_functions)

# Consensus method - only pass if non-default
if consensus_method and consensus_method != "rbr":
    command_list.extend(["--consensus_method", consensus_method])

# Pocket-specific arguments
if pocket_mode_display == "Custom" and mode == "single":
    pocket_str = "*".join([f"{k}:{','.join(map(str, v))}" for k, v in pocket_coordinates.items()])
    command_list.extend(["--manual_pocket", pocket_str])
    command_list.extend(["--pocket_method", pocket_mode])
elif pocket_mode_display in ["Reference", "RoG"]:
    command_list.extend(["--pocket_method", pocket_mode])
    if int(pocket_radius) != 10:
        command_list.extend(["--pocket_radius", str(int(pocket_radius))])

    # Add reference ligands (only those that correspond to valid receptors)
    valid_refs = [ref for i, ref in enumerate(st.session_state.reference_ligand_paths)
                 if i < len(st.session_state.receptor_paths) and ref.strip() and st.session_state.receptor_paths[i].strip()]

    if valid_refs:
        command_list.append("--reference_ligand")
        command_list.extend(valid_refs)

elif pocket_mode_display == "Dogsitescorer":
    command_list.extend(["--pocket_method", pocket_mode])
    command_list.extend(["--dogsitescorer_method", dss_mode])

# Ensemble mode arguments
if mode == "ensemble":
    command_list.extend(["--ensemble_top_percent", str(ensemble_top_percent)])

# Decoy generation parameters
if gen_decoys:
    command_list.extend([
        "--generate_decoys", bool_to_str(gen_decoys),
        "--decoy_model", decoy_model,
        "--n_decoys", str(n_decoys),
        "--actives", active_ligands,
    ])

# Show command preview
with st.expander("Command Preview"):
    st.code(" ".join(command_list))

def run_dockm8(cmd_list):
    """Run dockm8 with the provided command and capture output to log file"""
    print("Running command:", cmd_list)
    with open("log.txt", "w") as log_file:
        log_file.write(f"Command: {' '.join(cmd_list)}\n\n")

    process = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Non-blocking reading of process output
    def monitor_output():
        with open("log.txt", "a") as log_file:
            for line in iter(process.stdout.readline, ''):
                log_file.write(line)
                log_file.flush()

    import threading
    output_thread = threading.Thread(target=monitor_output)
    output_thread.daemon = True
    output_thread.start()

    # Store the process in session state for potential future handling
    st.session_state['dockm8_process'] = process

def read_log_file(file_path):
    """Read the content of the log file"""
    try:
        with open(file_path) as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading log file: {str(e)}"

# Collect all validation errors
all_errors = []
if software_error:
    all_errors.append(software_error)
all_errors.extend(receptor_errors)
all_errors.extend(reference_ligand_errors)
if ligand_file_error:
    all_errors.append(ligand_file_error)
if custom_pocket_ensemble_error and pocket_mode_display == "Custom" and mode == "ensemble":
    all_errors.append(custom_pocket_ensemble_error)
if plants_error and "PLANTS" in docking_programs:
    all_errors.append(plants_error)
if gen_decoys and active_ligands_error:
    all_errors.append(active_ligands_error)
if gen_decoys and mode == "ensemble":
    all_errors.append("Decoy generation is not supported in ensemble mode")
if not docking_programs:
    all_errors.append("At least one docking program must be selected")
if not rescoring_functions:
    all_errors.append("At least one scoring function must be selected")
if not selection_methods:
    all_errors.append("At least one pose selection method must be selected")

# Run the script file
if st.button("Run DockM8"):
    # Special validation for Reference/RoG without reference ligands
    if pocket_mode_display in ["Reference", "RoG"] and not any(p.strip() for p in st.session_state.reference_ligand_paths):
        st.error(f"'{pocket_mode_display}' pocket mode requires at least one reference ligand.")
    # Check if any valid receptors were provided
    elif not any(p.strip() for p in st.session_state.receptor_paths):
        st.error("At least one valid receptor path is required.")
    # Check ensemble mode requirements
    elif mode == "ensemble" and len([p for p in st.session_state.receptor_paths if p.strip()]) < 2:
        st.error("Ensemble mode requires at least two valid receptor paths.")
    # Check for any validation errors
    elif all_errors:
        st.error(f"Please fix the following errors before running:\n" + "\n".join(all_errors))
    # All validation passed, run the command
    else:
        run_dockm8(command_list)
        st.success("DockM8 is now running. Check the log below for progress updates.")

# Display log section
st.header("Log Output", divider="orange")
log_file_path = Path(CWD, "log.txt")

if log_file_path.exists():
    log_content = read_log_file(log_file_path)
    # Create an empty container for dynamic content updates
    log_container = st.empty()
    # Display initial log content
    log_container.text_area(
        "Log (refreshes automatically)",
        log_content,
        height=400
    )

    # Auto-refresh log using st.rerun with placeholder
    if st.checkbox("Auto-refresh log (every 5 seconds)", value=True):
        time.sleep(5)
        st.rerun()
else:
    st.info("No log file found. Run DockM8 to generate logs.")
