import sys
from pathlib import Path
import os
import traceback
import pandas as pd
import streamlit as st

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import PAGES, menu
from scripts.consensus.consensus import CONSENSUS_METHODS, apply_consensus_methods

menu()

st.title("Consensus")

# Check for prepared docking library
if 'rescored_poses' not in st.session_state:
	default_path_library = Path(
		st.session_state.w_dir
	) / "rescored_poses.sdf" if 'w_dir' in st.session_state else dockm8_path / "tests" / "test_files" / "allposes.sdf"
	library_to_rescore_input = st.text_input(label="Enter the path to the rescored poses file (.sdf format)",
												value=default_path_library,
												help="Choose a file containing rescored poses (.sdf format)")
	if not Path(library_to_rescore_input).is_file():
		st.error("File does not exist.")
	else:
		st.session_state.rescored_poses = Path(library_to_rescore_input)
		st.success(f"Library loaded: {library_to_rescore_input}")

st.subheader("Consensus Algorithm", divider="orange")
st.session_state.consensus_methods = st.selectbox(label="Choose which consensus algorithm to use:",
													index=9,
													options=list(CONSENSUS_METHODS.keys()),
													help="The method to use for consensus.")

st.subheader("Run Consensus Analysis", divider="orange")


def determine_working_directory() -> Path:
	if 'w_dir' in st.session_state:
		consensus_results_save_path = Path(st.session_state.w_dir)
		return consensus_results_save_path
	elif isinstance(st.session_state.rescored_poses, Path):
		consensus_results_save_path = st.session_state.rescored_poses.parent
		return consensus_results_save_path
	elif isinstance(st.session_state.rescored_poses, pd.DataFrame):
		custom_dir = st.text_input("Enter a custom save location:")
		if "." in custom_dir:
			st.error("Please enter a valid directory, not a file.")
		else:
			consensus_results_save_path = Path(custom_dir)
			consensus_results_save_path.mkdir(exist_ok=True, parents=True)
			return consensus_results_save_path
	st.error(
		"Unable to determine working directory. Please set a working directory or use a file path for the library.")
	return None


def run_consensus():
	common_params = {
		"poses_input": st.session_state.rescored_poses,
		"consensus_methods": st.session_state.consensus_methods,
		"standardization_type": "min_max"}
	if st.session_state.save_consensus_results:
		output_file = st.session_state.consensus_scores
		apply_consensus_methods(**common_params, output_path=output_file)
		st.session_state.consensus_scores = output_file
	else:
		results = apply_consensus_methods(**common_params)
		st.session_state.consensus_scores = results


col1, col2 = st.columns(2)
st.session_state.save_consensus_results = col2.toggle(label="Save Consensus scores",
														value=True,
														key='save_consensus_results_toggle')

if st.session_state.save_consensus_results:
	consensus_score_save_path = determine_working_directory()
	if consensus_score_save_path:
		st.session_state.consensus_scores = consensus_score_save_path
		col2.write(f'Consensus scores will be saved to: **{consensus_score_save_path}**')

if col1.button("Run Consensus scoring", use_container_width=True):
	if not st.session_state.consensus_methods:
		st.error("Please select at least one rescoring function.")
	else:
		try:
			st.write(st.session_state.rescored_poses)
			run_consensus()
			st.success("Rescoring completed successfully.")
		except Exception as e:
			st.error(f"An error occurred during rescoring: {str(e)}")
			st.error(traceback.format_exc())

if st.button('Generate DockM8 Report'):
	st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[10]))
