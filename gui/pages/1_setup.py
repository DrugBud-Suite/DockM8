import sys
from pathlib import Path
import os

import streamlit as st

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import menu

menu()

st.title("Setup")
CWD = os.getcwd()

mode = st.selectbox(
	label="Which mode do you want to run DockM8 in?",
	key=0,
	options=("Single", "Ensemble"),
	help="Single Docking: DockM8 will dock each ligand in the library to a single receptor. " +
	"Ensemble Docking: DockM8 will dock each ligand in the library to all specified receptors and then combine the results to create an ensemble consensus."
)

if mode != "Single":
	threshold = st.slider(
		label="Threshold for ensemble consensus (in %)",
		min_value=0.0,
		max_value=10.0,
		step=0.1,
		value=1.0,
		help=
		"Threshold for ensemble consensus (in %). DockM8 will only consider a ligand as a consensus hit if it is a top scorer for all the receptors."
	)

else:
	threshold = None

n_cpus = st.slider("Number of CPUs",
					min_value=1,
					max_value=os.cpu_count(),
					step=1,
					value=int(os.cpu_count() * 0.9),
					help="Number of CPUs to use for calculations")

software = st.text_input(
	"Choose a software directory",
	value=CWD + "/software",
	help="Type the directory containing the software folder: For example: /home/user/Dockm8/software")

w_dir = st.text_input(
	"Choose a working directory",
	value=CWD + "/tests/test_files/working_dir",
	help="Type the directory where DockM8 will store the results: For example: /home/user/Dockm8/working_dir")

if st.button("Confirm Settings"):
	st.session_state["mode"] = mode
	st.session_state["threshold"] = threshold
	st.session_state["n_cpus"] = n_cpus
	st.session_state["software"] = software
	st.session_state["w_dir"] = w_dir