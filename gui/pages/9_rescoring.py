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
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS, rescore_poses

menu()

if 'rescoring_functions' not in st.session_state:
	st.session_state.rescoring_functions = None

st.title("Rescoring")

st.subheader("Scoring Functions", divider="orange")

selected_functions = []

cols = st.columns(3)
for i, (function_name, function_class) in enumerate(RESCORING_FUNCTIONS.items()):
	with cols[i % 3]:
		if cols[i % 3].toggle(function_name, key=f"checkbox_{function_name}"):
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
if st.button("Run Rescoring"):
	with st.spinner("Running rescoring..."):
		if not selected_functions:
			st.warning("Please select at least one scoring function.")
		elif 'w_dir' not in st.session_state or 'prepared_protein_path' not in st.session_state:
			st.error("Please complete the previous steps before running rescoring.")
		else:
			try:
				w_dir = st.session_state.w_dir
				protein_file = st.session_state.prepared_protein_path
				software = st.session_state.software if 'software' in st.session_state else Path(dockm8_path /
																									'software')
				n_cpus = st.session_state.n_cpus if 'n_cpus' in st.session_state else os.cpu_count()
				pocket_definition = st.session_state.pocket_definition if 'pocket_definition' in st.session_state else None

				clustered_sdf = w_dir / "clustering" / f"{st.session_state.pose_selection_method}_clustered.sdf"

				rescore_poses(w_dir=w_dir,
								protein_file=protein_file,
								pocket_definition=pocket_definition,
								software=software,
								clustered_sdf=clustered_sdf,
								functions=selected_functions,
								n_cpus=n_cpus)
				st.success(f"Rescoring completed successfully for functions: {', '.join(selected_functions)}")
			except Exception as e:
				st.error(f"An error occurred during rescoring: {str(e)}")
				st.error(traceback.format_exc())
	if st.button('Proceed to Consensus'):
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[10]))
