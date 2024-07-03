import sys
from pathlib import Path
import os
import traceback

import streamlit as st

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import PAGES, menu
from scripts.consensus.consensus import CONSENSUS_METHODS, apply_consensus_methods
from scripts.pose_selection.pose_selection import select_poses
from scripts.utilities.utilities import parallel_SDF_loader

menu()

st.title("Consensus")

st.subheader("Consensus Algorithm", divider="orange")
consensus_method = st.selectbox(label="Choose which consensus algorithm to use:",
								index=9,
								options=list(CONSENSUS_METHODS.keys()),
								help="The method to use for consensus.")

st.subheader("Run Consensus Analysis", divider="orange")
if st.button("Run Consensus Analysis"):
	with st.spinner("Running consensus analysis..."):
		if 'w_dir' not in st.session_state or 'prepared_protein_path' not in st.session_state:
			st.error("Please complete the previous steps before running consensus analysis.")
		else:
			w_dir = st.session_state.w_dir
			protein_file = st.session_state.prepared_protein_path
			software = st.session_state.software if 'software' in st.session_state else Path(dockm8_path / 'software')
			n_cpus = st.session_state.n_cpus if 'n_cpus' in st.session_state else os.cpu_count()

			try:
				# Assuming the pose selection and rescoring steps have been completed
				selection_methods = st.session_state.get('pose_selection_methods')
				rescoring_functions = st.session_state.get('rescoring_functions')

				for method in selection_methods:
					apply_consensus_methods(w_dir=w_dir,
											selection_method=method,
											consensus_methods=consensus_method,
											rescoring_functions=rescoring_functions,
											standardization_type="min_max"                            # You might want to make this configurable
											)

				st.success(f"Consensus analysis completed successfully using {consensus_method} method.")
			except Exception as e:
				st.error(f"An error occurred during consensus analysis: {str(e)}")
				st.error(traceback.format_exc())
	if st.button('Generate DockM8 Report'):
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[10]))
