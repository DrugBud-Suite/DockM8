import streamlit as st
from pathlib import Path
import sys

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import menu
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS
from scripts.pose_selection.clustering.clustering_metrics.clustering_metrics import CLUSTERING_METRICS

menu()

def app():
	st.title("Pose Selection")

	pose_selection = st.multiselect(
		label="Choose the pose selection method you want to use",
		default=["KORP-PL"],
		options=list(CLUSTERING_METRICS.keys()) +
		["bestpose", "bestpose_GNINA", "bestpose_SMINA", "bestpose_PLANTS", "bestpose_QVINA2", "bestpose_QVINAW"] +
		list(RESCORING_FUNCTIONS.keys()),
		help="The method(s) to use for pose clustering. Must be one or more of:\n" +
		"- RMSD : Cluster compounds on RMSD matrix of poses \n" +
		"- spyRMSD : Cluster compounds on symmetry-corrected RMSD matrix of poses\n" +
		"- espsim : Cluster compounds on electrostatic shape similarity matrix of poses\n" +
		"- USRCAT : Cluster compounds on shape similarity matrix of poses\n" +
		"- 3DScore : Selects pose with the lowest average RMSD to all other poses\n" +
		"- bestpose : Takes the best pose from each docking program\n" +
		"- bestpose_GNINA : Takes the best pose from GNINA docking program\n" +
		"- bestpose_SMINA : Takes the best pose from SMINA docking program\n" +
		"- bestpose_QVINAW : Takes the best pose from QVINAW docking program\n" +
		"- bestpose_QVINA2 : Takes the best pose from QVINA2 docking program\n" +
		"- bestpose_PLANTS : Takes the best pose from PLANTS docking program  \n" +
		"- You can also use any of the scoring functions and DockM8 will select the best pose for each compound according to the specified scoring function."
	)

	# Clustering algorithm
	if any(x in CLUSTERING_METRICS.keys() for x in pose_selection):
		clustering_algorithm = st.selectbox(
			label="Which clustering algorithm do you want to use?",
			options=("KMedoids", "Aff_Prop"),
			index=0,
			help=
			'Which algorithm to use for clustering. Must be one of "KMedoids", "Aff_prop". Must be set when using "RMSD", "spyRMSD", "espsim", "USRCAT" clustering metrics.'
		)

	else:
		clustering_algorithm = None

app()
