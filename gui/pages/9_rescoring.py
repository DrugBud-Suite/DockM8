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
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS, rescore_poses

menu()

# Check for prepared docking library
if 'selected_poses' not in st.session_state:
	default_path_library = Path(
		st.session_state.w_dir
	) / "selected_poses.sdf" if 'w_dir' in st.session_state else dockm8_path / "tests" / "test_files" / "allposes.sdf"
	library_to_rescore_input = st.text_input(label="Enter the path to the ligand library file (.sdf format)",
		value=default_path_library,
		help="Choose a ligand library file (.sdf format)")
	if not Path(library_to_rescore_input).is_file():
		st.error("File does not exist.")
	else:
		st.session_state.poses_to_rescore = library_to_rescore_input
		st.success(f"Library loaded: {library_to_rescore_input}")

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
	st.session_state.pose_selection_methods = None

st.title("Rescoring")

st.subheader("Scoring Functions", divider="orange")

st.warning(
	"Disclaimer: Slower scoring functions are not necessarily more accurate. The choice of scoring function should be based on your specific use case and requirements."
)

# Categorize scoring functions
scoring_categories = {
	"Empirical": {
	"Faster": ["CHEMPLP", "PLP", "Vinardo", "LinF9"], "Slower": ["GNINA-Affinity", "AAScore"]},
	"Semi-Empirical": {
	"Faster": ["AD4"], "Slower": []},
	"Knowledge-based": {
	"Faster": ["KORP-PL", "DLIGAND2", "ITScoreAff", "ConvexPLR"], "Slower": []},
	"Shape Similarity": {
	"Faster": ["PANTHER", "PANTHER-ESP", "PANTHER-Shape"], "Slower": []},
	"Machine Learning": {
	"Faster": ["GenScore-scoring", "GenScore-docking", "GenScore-balanced"],
	"Slower": ["RFScoreVS", "SCORCH", "CENsible", "RTMScore", "PLECScore", "NNScore"]}}

selected_functions = []

# Filter out empty categories
non_empty_categories = {
	category: speed_dict
	for category,
	speed_dict in scoring_categories.items()
	if any(func for speed, funcs in speed_dict.items() for func in funcs if func in RESCORING_FUNCTIONS)}

# Create columns for each non-empty category
cols = st.columns(len(non_empty_categories))

for i, (category, speed_dict) in enumerate(non_empty_categories.items()):
	with cols[i]:
		st.markdown(f"**{category}**")

		for speed in ["Faster", "Slower"]:
			functions = [func for func in speed_dict[speed] if func in RESCORING_FUNCTIONS]
			if functions:
				st.markdown(f"*{speed}*")
				for function_name in functions:
					if st.toggle(function_name, key=f"toggle_{speed}_{function_name}"):
						selected_functions.append(function_name)

st.write("Selected scoring functions:", ", ".join(selected_functions) if selected_functions else "None")

st.session_state.rescoring_functions = selected_functions

st.subheader("Score Manipulation", divider="orange")
normalize_scores = st.toggle(label="Normalize scores",
	value=True,
	help="Normalize scores to a range of 0-1",
	key="normalize_scores")
mw_scores = st.toggle(label="Normalize to MW",
	value=False,
	help="Scale the scores to molecular weight of the compound",
	key="mw_scores")

st.subheader("Run Rescoring", divider="orange")


def determine_working_directory() -> Path:
	if 'w_dir' in st.session_state:
		rescored_poses_save_path = Path(st.session_state.w_dir) / "rescored_poses.sdf"
		return rescored_poses_save_path
	elif isinstance(st.session_state.selected_poses, Path):
		rescored_poses_save_path = st.session_state.selected_poses.parent / "rescored_poses.sdf"
		return rescored_poses_save_path
	elif isinstance(st.session_state.selected_poses, pd.DataFrame):
		custom_dir = st.text_input("Enter a custom save location:")
		if custom_dir.endswith(".sdf"):
			rescored_poses_save_path = Path(custom_dir)
			rescored_poses_save_path.parent.mkdir(exist_ok=True, parents=True)
			return rescored_poses_save_path
		elif "." in custom_dir and custom_dir.split(".")[-1] != "sdf":
			st.error("Please enter a valid .sdf file path or a directory.")
		else:
			rescored_poses_save_path = Path(custom_dir) / "rescored_poses.sdf"
			rescored_poses_save_path.parent.mkdir(exist_ok=True, parents=True)
			return rescored_poses_save_path
	st.error(
		"Unable to determine working directory. Please set a working directory or use a file path for the library.")
	return None


def run_rescoring():
	common_params = {
		"poses": st.session_state.selected_poses,
		"protein_file": Path(st.session_state.prepared_protein_path),
		"pocket_definition": st.session_state.binding_site,
		"software": st.session_state.software,
		"functions": st.session_state.rescoring_functions,
		"n_cpus": st.session_state.get('n_cpus', int(os.cpu_count() * 0.9)), }
	if st.session_state.save_rescoring_results:
		output_file = st.session_state.rescored_poses
		rescore_poses(**common_params, output_file=output_file)
		st.session_state.rescored_poses = output_file
	else:
		results = rescore_poses(**common_params)
		st.session_state.rescored_poses = results


col1, col2 = st.columns(2)
st.session_state.save_rescoring_results = col2.toggle(label="Save Rescored Poses to SDF file",
	value=True,
	key='save_rescoring_results_toggle')

if st.session_state.save_rescoring_results:
	rescored_poses_save_path = determine_working_directory()
	if rescored_poses_save_path:
		st.session_state.rescored_poses = rescored_poses_save_path
		col2.write(f'Rescored poses will be saved to: **{rescored_poses_save_path}**')

if col1.button("Run Rescoring", use_container_width=True):
	if not st.session_state.rescoring_functions:
		st.error("Please select at least one rescoring function.")
	else:
		try:
			run_rescoring()
			st.success("Rescoring completed successfully.")
		except Exception as e:
			st.error(f"An error occurred during rescoring: {str(e)}")
			st.error(traceback.format_exc())

if st.button('Proceed to Consensus Scoring'):
	if 'rescored_poses' in st.session_state:
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[9]))
	else:
		st.warning("Postprocessing was skipped, proceeding with docked poses")
		st.session_state.rescored_poses = st.session_state.selected_poses
