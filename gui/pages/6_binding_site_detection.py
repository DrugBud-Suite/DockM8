import streamlit as st
from pathlib import Path
import sys

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import menu
from scripts.pocket_finding.pocket_finding import POCKET_DETECTION_OPTIONS

menu()

def app():
	st.title("Binding Site Detection")
	# Pocket finding
	st.header("Binding Pocket definition", divider="orange")
	pocket_mode = st.selectbox(
		label="How should the pocket be defined?",
		options=POCKET_DETECTION_OPTIONS,
		help="Reference Ligand: DockM8 will use the reference ligand to define the pocket. " +
		"RoG: DockM8 will use the reference ligand radius of gyration. " +
		"DogSiteScorer: DockM8 will use the DogSiteScorer pocket finding algorithm to define the pocket." +
		"P2Rank: DockM8 will use the P2Rank pocket finding algorithm to define the pocket." +
		"Manual: Define your own pocket center and size coordinates.")

	if pocket_mode in ("Reference", "RoG"):
		pocket_radius = st.number_input("Binding Site Radius", min_value=0.0, value=10.0, step=0.1)
		reference_files = st.text_input(
			label="File path(s) of one or more multiple reference ligand files (.sdf format), separated by commas",
			help="Choose one or multiple reference ligand files (.pdb format)",
			value=str(dockm8_path / "tests" / "test_files" / "1fvv_l.sdf"),)

		reference_files = [Path(file.strip()) for file in reference_files.split(",")]
		# Reference files validation
		for file in reference_files:
			if not Path(file).is_file():
				st.error(f"Invalid file path: {file}")
		x_center = y_center = z_center = x_size = y_size = z_size = manual_pocket = None
	# Custom pocket
	elif pocket_mode == "Manual" and st.session_state["mode"] == "Single":
		col1, col2, col3 = st.columns(3)
		x_center = col1.number_input(label="X Center", value=0.0, help="Enter the X coordinate of the pocket center")
		y_center = col2.number_input(label="Y Center", value=0.0, help="Enter the Y coordinate of the pocket center")
		z_center = col3.number_input(label="Z Center", value=0.0, help="Enter the Z coordinate of the pocket center")
		x_size = col1.number_input(label="X Size",
									value=20.0,
									help="Enter the size of the pocket in the X direction (in Angstroms)")
		y_size = col2.number_input(label="Y Size",
									value=20.0,
									help="Enter the size of the pocket in the Y direction (in Angstroms)")
		z_size = col3.number_input(label="Z Size",
									value=20.0,
									help="Enter the size of the pocket in the Z direction (in Angstroms)")
		manual_pocket = f"center:{x_center},{y_center},{z_center}*size:{x_size},{y_size},{z_size}"
		pocket_radius = reference_files = None
	elif pocket_mode == "Manual" and st.session_state["mode"] != "Single":
		st.error(
			"Manual pocket definition does not currently work in ensemble mode, please change the pocket definition mode")
	else:
		pocket_radius = reference_files = x_center = y_center = z_center = x_size = y_size = z_size = manual_pocket = None


app()
