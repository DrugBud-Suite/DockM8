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
from gui.utils import save_dataframe_to_sdf
from scripts.docking_postprocessing.docking_postprocessing import docking_postprocessing

menu()

st.title("Docking Postprocessing")

# Check for prepared docking library
if 'poses_for_postprocessing' not in st.session_state:
	default_path_library = Path(
		st.session_state.w_dir
	) / "docking/allposes.sdf" if 'w_dir' in st.session_state else dockm8_path / "tests" / "test_files" / "allposes.sdf"
	library_to_postprocess_input = st.text_input(label="Enter the path to the docked ligands (.sdf format)",
													value=default_path_library,
													help="Choose a file containing docked ligands (.sdf format)")
	if not Path(library_to_postprocess_input).is_file():
		st.error("File does not exist.")
	else:
		st.session_state.poses_for_postprocessing = library_to_postprocess_input
		st.success(f"Library loaded: {library_to_postprocess_input}")

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

if 'software' not in st.session_state:
	st.info("Software path not set. Using default path. If you want to change it you can do so on the 'Setup' page.")
	st.session_state.software = dockm8_path / "software"

st.subheader("Pose Minimization", divider="orange")
col1, col2 = st.columns(2)
minimize_poses = col1.toggle(label="Minimize docking poses",
								value=False,
								help="Minimize poses after docking",
								key="minimize_poses")
if minimize_poses:
	minimize_poses_config = {}
	minimize_poses_config['force_field'] = col1.selectbox(label="Choose the force field for minimization",
															options=("UFF", "MMFF94", "MMFF94s"),
															index=2)
	minimize_poses_config['n_steps'] = col1.number_input(label="Number of minimization steps",
															min_value=1,
															max_value=5000,
															value=1000,
															step=10,
															help="Number of minimization steps to perform")
	minimize_poses_config['distance_constraing'] = col1.number_input(
		label="Distance constraint (Å)",
		min_value=0.1,
		max_value=2.0,
		value=1.0,
		step=0.1,
		help="Distance constraint in Å for minimization")
minimize_poses_H = col2.toggle(label="Minimize Hydrogens of docking poses",
								value=False,
								help="Minimize H positions of poses after docking",
								key="minimize_poses_H")

st.subheader("Clash and Strain Filtering", divider="orange")
col1, col2 = st.columns(2)
clash_cutoff_toggle = col1.toggle(label="Remove poses with clashes",
									value=True,
									help="Remove poses with clashes",
									key="clash_cutoff_toggle")
if clash_cutoff_toggle:
	clash_cutoff = col1.number_input(label="Remove poses with more than x clashes:",
										min_value=0,
										max_value=100,
										value=5,
										step=1,
										help="Setting too low will remove too many poses, use default if unsure")
else:
	clash_cutoff = None

strain_cutoff_toggle = col2.toggle(label="Remove poses with high strain",
									value=True,
									help="Remove poses with high strain",
									key="strain_cutoff_toggle")
if strain_cutoff_toggle:
	strain_cutoff = col2.number_input(label="Remove poses with higher than x strain energy (kcal/mol):",
										min_value=100,
										max_value=100000,
										value=5000,
										step=100,
										help="Setting too low will remove too many poses, use default if unsure")
else:
	strain_cutoff = None

st.subheader("PoseBusters Analysis", divider="orange")
bust_poses = st.toggle(
	label="Bust poses using PoseBusters",
	value=True,
	help=
	"Bust poses using PoseBusters: Will remove any poses with clashes, non-flat aromatic rings etc. WARNING: may take a long time to run",
	key="bust_poses_toggle")

st.subheader("Classy_Pose Classification", divider="orange")
classy_pose = st.toggle(label="Classify poses using Classy_Pose",
						value=False,
						help="Classify poses using Classy_Pose",
						key="classy_pose_toggle")
if classy_pose:
	classy_pose_model = st.selectbox(label="Choose the Classy_Pose model to use",
										options=("SVM (from publication)", "LGBM (retrained model)"),
										index=0)

st.subheader("Run Postprocessing", divider="orange")


def determine_working_directory() -> Path:
	if 'w_dir' in st.session_state:
		postprocessed_poses_save_path = Path(st.session_state.w_dir) / "postprocessed_poses.sdf"
		return postprocessed_poses_save_path
	elif isinstance(st.session_state.poses_for_postprocessing, Path):
		postprocessed_poses_save_path = st.session_state.poses_for_postprocessing.parent / "postprocessed_poses.sdf"
		return postprocessed_poses_save_path
	elif isinstance(st.session_state.poses_for_postprocessing, pd.DataFrame):
		custom_dir = st.text_input("Enter a custom save location:")
		# If user enters a file path
		if custom_dir.endswith(".sdf"):
			postprocessed_poses_save_path = Path(custom_dir)
			postprocessed_poses_save_path.parent.mkdir(exist_ok=True, parents=True)
			return postprocessed_poses_save_path
		elif "." in custom_dir and custom_dir.split(".")[-1] != "sdf":
			st.error("Please enter a valid .sdf file path or a directory.")
		# If user enters a directory path
		else:
			postprocessed_poses_save_path = Path(custom_dir) / "postprocessed_poses.sdf"
			postprocessed_poses_save_path.parent.mkdir(exist_ok=True, parents=True)
			return postprocessed_poses_save_path
	st.error(
		"Unable to determine working directory. Please set a working directory or use a file path for the library.")
	return None


def run_postprocessing():
	common_params = {
		"input_data": st.session_state.poses_for_postprocessing,
		"protein_file": Path(st.session_state.prepared_protein_path),
		"minimize_poses": minimize_poses,
		"bust_poses": bust_poses,
		"strain_cutoff": strain_cutoff,
		"clash_cutoff": clash_cutoff,
		"classy_pose": classy_pose,
		"classy_pose_model": classy_pose_model if classy_pose else None,
		"n_cpus": st.session_state.get('n_cpus', int(os.cpu_count() * 0.9)), }
	if st.session_state.save_postprocessing_results:
		output_sdf = st.session_state.poses_for_selection
		docking_postprocessing(**common_params, output_sdf=output_sdf)
		st.session_state.poses_for_selection = output_sdf

	else:
		results = docking_postprocessing(**common_params)
		st.session_state.poses_for_selection = results


# UI Layout
col1, col2 = st.columns(2)
st.session_state.save_postprocessing_results = col2.toggle(label="Save Postprocessed Results to SDF file",
															value=True,
															key='save_postprocessing_results_toggle')

if st.session_state.save_docking_results:
	# Determine and set working directory
	postprocessed_poses_save_path = determine_working_directory()
	if postprocessed_poses_save_path:
		st.session_state.poses_for_selection = postprocessed_poses_save_path

if st.button("Run Docking Postprocessing"):
	try:
		run_postprocessing()
		st.success("Docking postprocessing completed successfully.")
	except Exception as e:
		st.error(f"An error occurred during docking postprocessing: {str(e)}")
		st.error(traceback.format_exc())

if st.button('Proceed to Pose Selection'):
	if 'poses_for_selection' in st.session_state:
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[7]))
	else:
		st.warning("Postprocessing was skipped, proceeding with docked poses")
		st.session_state.poses_for_selection = st.session_state.poses_for_postprocessing
