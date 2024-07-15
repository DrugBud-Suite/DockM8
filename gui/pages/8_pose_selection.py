import os
import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import PAGES, menu
from scripts.pose_selection.clustering.clustering_metrics.clustering_metrics import CLUSTERING_METRICS
from scripts.pose_selection.pose_selection import select_poses
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS

menu()

# Check for prepared docking library
if 'poses_for_selection' not in st.session_state:
	default_path_library = Path(
		st.session_state.w_dir
	) / "postprocessed_poses.sdf" if 'w_dir' in st.session_state else dockm8_path / "tests" / "test_files" / "postprocessed_poses.sdf"
	library_to_select_input = st.text_input(label="Enter the path to the ligand library file (.sdf format)",
				value=default_path_library,
				help="Choose a ligand library file (.sdf format)")
	if not Path(library_to_select_input).is_file():
		st.error("File does not exist.")
	else:
		st.session_state.poses_for_selection = library_to_select_input
		st.success(f"Library loaded: {library_to_select_input}")

# Check for prepared protein file
if 'prepared_protein_path' not in st.session_state:
	default_path_protein = Path(
		st.session_state.w_dir
	) / "prepared_protein.pdb" if 'w_dir' in st.session_state else dockm8_path / "tests" / "test_files" / "prepared_protein.pdb"
	st.warning("Prepared Protein File is missing.")
	protein_path = st.text_input("Enter the path to the prepared protein file (.pdb):",
			value=default_path_protein,
			help="Enter the complete file path to your prepared protein file.")
	if not Path(protein_path).is_file():
		st.error("File does not exist.")
	else:
		st.session_state.prepared_protein_path = protein_path
		st.success(f"Protein file loaded: {protein_path}")

# Check for binding site definition
if 'binding_site' not in st.session_state:
	st.warning("Binding Site Definition is missing.")
	if st.button("Define Binding Site"):
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[4]))

if 'software' not in st.session_state:
	st.info("Software path not set. Using default path. If you want to change it you can do so on the 'Setup' page.")
	st.session_state.software = dockm8_path / "software"

if 'pose_selection_method' in st.session_state:
	st.session_state.pose_selection_method = None


def group_methods():
	return {
		"Clustering Methods": list(CLUSTERING_METRICS.keys()),
		"Best Pose Methods": [
		"bestpose", "bestpose_GNINA", "bestpose_SMINA", "bestpose_PLANTS", "bestpose_QVINA2", "bestpose_QVINAW"],
		"Rescoring Functions": list(RESCORING_FUNCTIONS.keys())}


st.title("Pose Selection")

method_groups = group_methods()
selected_method = []

# Custom CSS to increase font size of expander headers
st.markdown("""
    <style>
    .streamlit-expanderHeader {
        font-size: 1.2em !important;
        font-weight: bold !important;
    }
    </style>
    """,
	unsafe_allow_html=True)

for group, methods in method_groups.items():
	st.subheader(group, divider="orange")
	# Create three columns
	cols = st.columns(3)

	for i, method in enumerate(methods):
		disabled = False
		if group == "Best Pose Methods" and method != "bestpose":
			docking_program = method.split("_")[1]
			if 'docking_programs' in st.session_state and docking_program not in st.session_state.docking_programs:
				disabled = True

		# Use the appropriate column (i % 3 cycles through 0, 1, 2)
		with cols[i % 3]:
			if disabled:
				cols[i % 3].toggle(method, key=f"checkbox_{method}", disabled=True)
			else:
				if cols[i % 3].toggle(method, key=f"checkbox_{method}"):
					selected_method.append(method)

	if group == "Best Pose Methods":
		st.write("Note: Best pose methods are disabled for docking programs that were not used in the previous step.")

st.session_state.pose_selection_method = selected_method

# Clustering Algorithm section
if any(x in CLUSTERING_METRICS.keys() for x in selected_method):
	st.subheader("Clustering Algorithm", divider="orange")
	clustering_algorithm = st.selectbox(
		label="Which clustering algorithm do you want to use?",
		options=("KMedoids", "Aff_Prop"),
		index=0,
		help='Which algorithm to use for clustering. Must be set when using clustering metrics.')
	st.session_state.clustering_algorithm = clustering_algorithm
else:
	st.session_state.clustering_algorithm = None

# Run Pose Selection section
st.subheader("Run Pose Selection", divider="orange")


def determine_working_directory() -> Path:
	if 'w_dir' in st.session_state:
		selected_poses_save_path = Path(st.session_state.w_dir) / "selected_poses.sdf"
		return selected_poses_save_path
	elif isinstance(st.session_state.poses_for_selection, Path):
		selected_poses_save_path = st.session_state.poses_for_selection.parent / "selected_poses.sdf"
		return selected_poses_save_path
	elif isinstance(st.session_state.poses_for_selection, pd.DataFrame):
		custom_dir = st.text_input("Enter a custom save location:")
		if custom_dir.endswith(".sdf"):
			selected_poses_save_path = Path(custom_dir)
			selected_poses_save_path.parent.mkdir(exist_ok=True, parents=True)
			return selected_poses_save_path
		elif "." in custom_dir and custom_dir.split(".")[-1] != "sdf":
			st.error("Please enter a valid .sdf file path or a directory.")
		else:
			selected_poses_save_path = Path(custom_dir) / "selected_poses.sdf"
			selected_poses_save_path.parent.mkdir(exist_ok=True, parents=True)
			return selected_poses_save_path
	st.error(
		"Unable to determine working directory. Please set a working directory or use a file path for the library.")
	return None


def run_pose_selection():
	common_params = {
		"poses": st.session_state.poses_for_selection,
		"selection_method": st.session_state.pose_selection_method,
		"clustering_method": st.session_state.clustering_algorithm,
		"pocket_definition": st.session_state.binding_site,
		"protein_file": Path(st.session_state.prepared_protein_path),
		"software": st.session_state.software,
		"n_cpus": st.session_state.get('n_cpus', int(os.cpu_count() * 0.9)), }
	if st.session_state.save_selection_results:
		output_file = st.session_state.selected_poses
		select_poses(**common_params, output_file=output_file)
		st.session_state.selected_poses = output_file
	else:
		results = select_poses(**common_params)
		st.session_state.selected_poses = results


col1, col2 = st.columns(2)
st.session_state.save_selection_results = col2.toggle(label="Save Selected Poses to SDF file",
				value=True,
				key='save_selection_results_toggle')

if st.session_state.save_selection_results:
	selected_poses_save_path = determine_working_directory()
	if selected_poses_save_path:
		st.session_state.selected_poses = selected_poses_save_path
		col2.write(f'Selected poses will be saved to: **{selected_poses_save_path}**')

if st.button("Run Pose Selection"):
	try:
		run_pose_selection()
		st.success("Pose selection completed successfully.")
	except Exception as e:
		st.error(f"An error occurred during pose selection: {str(e)}")
		st.error(traceback.format_exc())

if st.button('Proceed to Rescoring'):
	if 'selected_poses' in st.session_state:
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[8]))
	else:
		st.session_state.selected_poses = st.session_state.poses_for_selection
		st.warning("Please run pose selection before proceeding to rescoring.")
