import os
import sys
import traceback
from pathlib import Path

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
from scripts.utilities.utilities import parallel_SDF_loader

menu()

if 'pose_selection_method' in st.session_state:
	st.session_state.pose_selection_methods = None

def group_methods():
	return {
		"Clustering Methods": list(CLUSTERING_METRICS.keys()),
		"Best Pose Methods": [
			"bestpose", "bestpose_GNINA", "bestpose_SMINA", "bestpose_PLANTS", "bestpose_QVINA2", "bestpose_QVINAW"],
		"Rescoring Functions": list(RESCORING_FUNCTIONS.keys())}


st.title("Pose Selection")

method_groups = group_methods()
selected_methods = []

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
					selected_methods.append(method)

	if group == "Best Pose Methods":
		st.write("Note: Best pose methods are disabled for docking programs that were not used in the previous step.")

st.write("Selected methods:", ", ".join(selected_methods) if selected_methods else "None")

st.session_state.pose_selection_methods = selected_methods

# Clustering Algorithm section
if any(x in CLUSTERING_METRICS.keys() for x in selected_methods):
	st.subheader("Clustering Algorithm", divider="orange")
	clustering_algorithm = st.selectbox(
		label="Which clustering algorithm do you want to use?",
		options=("KMedoids", "Aff_Prop"),
		index=0,
		help='Which algorithm to use for clustering. Must be set when using clustering metrics.')
else:
	clustering_algorithm = None

# Run Pose Selection section
st.subheader("Run Pose Selection", divider="orange")
if st.button("Run Pose Selection"):
	with st.spinner("Running pose selection..."):
		if not selected_methods:
			st.warning("Please select at least one pose selection method.")
		elif 'w_dir' not in st.session_state or 'prepared_protein_path' not in st.session_state:
			st.error("Please complete the previous steps before running pose selection.")
		else:
			w_dir = st.session_state.w_dir
			protein_file = st.session_state.prepared_protein_path
			software = st.session_state.software if 'software' in st.session_state else Path(dockm8_path / 'software')
			n_cpus = st.session_state.n_cpus if 'n_cpus' in st.session_state else os.cpu_count()

			try:
				all_poses_sdf = w_dir / "allposes.sdf"
				all_poses = parallel_SDF_loader(all_poses_sdf, molColName="Molecule", idName="Pose ID", n_cpus=n_cpus)

				for selection_method in selected_methods:
					select_poses(selection_method=selection_method,
						clustering_method=clustering_algorithm,
						w_dir=w_dir,
						protein_file=protein_file,
						software=software,
						all_poses=all_poses,
						n_cpus=n_cpus)
				st.success(f"Pose selection completed successfully for methods: {', '.join(selected_methods)}")
			except Exception as e:
				st.error(f"An error occurred during pose selection: {str(e)}")
				st.error(traceback.format_exc())
	if st.button('Proceed to Rescoring'):
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[9]))
