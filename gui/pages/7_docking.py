import streamlit as st
from pathlib import Path
import sys
import os

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import menu
from scripts.docking.docking import DOCKING_PROGRAMS

menu()

def app():
	st.title("Library Analysis and Filtering")

	# Docking programs
	st.header("Docking programs", divider="orange")
	docking_programs = st.multiselect(label="Choose the docking programs you want to use",
										default=["GNINA"],
										options=DOCKING_PROGRAMS,
										help="Choose the docking programs you want to use, mutliple selection is allowed")

	if "PLANTS" in docking_programs and not os.path.exists(f"{st.session_state['software']}/PLANTS"):
		st.warning(
			"PLANTS was not found in the software folder, please visit http://www.tcd.uni-konstanz.de/research/plants.php")

	# Number of poses
	n_poses = st.slider(label="Number of poses",
						min_value=1,
						max_value=100,
						step=5,
						value=10,
						help="Number of poses to generate for each ligand")

	# Exhaustiveness
	exhaustiveness = st.select_slider(
		label="Exhaustiveness",
		options=[1, 2, 4, 8, 16, 32],
		value=8,
		help=
		"Exhaustiveness of the docking, only applies to GNINA, SMINA, QVINA2 and QVINAW. Higher values can significantly increase the runtime."
	)

app()
