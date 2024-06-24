import os
import sys
from pathlib import Path

import streamlit as st
from streamlit_molstar import st_molstar
from streamlit_molstar.docking import st_molstar_docking
from streamlit_molstar.pocket import select_pocket_from_local_protein

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import PAGES, menu
from scripts.pocket_finding.default import find_pocket_default
from scripts.pocket_finding.dogsitescorer import find_pocket_dogsitescorer
from scripts.pocket_finding.manual import parse_pocket_coordinates
from scripts.pocket_finding.p2rank import download_p2rank, find_pocket_p2rank
from scripts.pocket_finding.pocket_finding import POCKET_DETECTION_OPTIONS
from scripts.pocket_finding.radius_of_gyration import find_pocket_RoG
from scripts.utilities.pocket_extraction import extract_pocket
from scripts.utilities.utilities import printlog

menu()

st.title("Binding Site Detection")

pocket_definition = None

if 'prepared_protein_path' not in st.session_state:
	protein_input = st.text_input("Enter file path (.pdb):",
									value=str(dockm8_path / "tests" / "test_files" / "1fvv_p.pdb"),
									help="Enter the complete file path to your protein data.")
else:
	protein_input = st.session_state.prepared_protein_path

st.header("Binding Pocket definition", divider="orange")
pocket_mode = st.selectbox(
	label="Select a binding site detection method:",
	options=POCKET_DETECTION_OPTIONS,
	help="Reference Ligand: DockM8 will use the reference ligand to define the pocket. " +
	"RoG: DockM8 will use the reference ligand radius of gyration. " +
	"DogSiteScorer: DockM8 will use the DogSiteScorer pocket finding algorithm to define the pocket." +
	"P2Rank: DockM8 will use the P2Rank pocket finding algorithm to define the pocket." +
	"Manual: Define your own pocket center and size coordinates.")

if pocket_mode == 'Reference':
	col1, col2 = st.columns(2)
	pocket_radius = col1.number_input("Binding Site Radius", min_value=0.0, value=10.0, step=0.1)
	reference_files = col2.text_input(
		label="File path(s) of one or more multiple reference ligand files (.sdf format), separated by commas",
		help="Choose one or multiple reference ligand files (.sdf format)",
		value=str(dockm8_path / "tests" / "test_files" / "1fvv_l.sdf"),
	)
	reference_files = [Path(file.strip()) for file in reference_files.split(",")]
	# Reference files validation
	for file in reference_files:
		if not Path(file).is_file():
			st.error(f"Invalid file path: {file}")

	if st.button("Find Pocket"):
		try:
			pocket_definition = find_pocket_default(reference_files[0], Path(protein_input), pocket_radius)
			st.success("Pocket found successfully!")
		except Exception as e:
			st.error(f"Error in finding pocket: {str(e)}")

elif pocket_mode == "RoG":
	reference_files = st.text_input(
		label="File path(s) of one or more multiple reference ligand files (.sdf format), separated by commas",
		help="Choose one or multiple reference ligand files (.sdf format)",
		value=str(dockm8_path / "tests" / "test_files" / "1fvv_l.sdf"),
	)
	reference_files = [Path(file.strip()) for file in reference_files.split(",")]
	# Reference files validation
	for file in reference_files:
		if not Path(file).is_file():
			st.error(f"Invalid file path: {file}")

	if st.button("Find Pocket"):
		try:
			pocket_definition = find_pocket_RoG(reference_files[0], Path(protein_input))
			st.success("Pocket found successfully!")
		except Exception as e:
			st.error(f"Error in finding pocket: {str(e)}")

elif pocket_mode == "Dogsitescorer":
	dogsitescorer_mode = st.selectbox(label="Choose which metric to select binding sites by:",
										options=["Volume", "Druggability_Score", "Surface", "Depth"],
										help="Choose the metric to select binding sites by.")
	if st.button("Find Pockets"):
		try:
			pocket_definition = find_pocket_dogsitescorer(Path(protein_input), method=dogsitescorer_mode)
			st.success("Pocket found successfully!")
		except Exception as e:
			st.error(f"Error in finding pocket: {str(e)}")

elif pocket_mode == "p2rank":
	pocket_radius = st.number_input("Binding Site Radius", min_value=0.0, value=10.0, step=0.1)
	software = st.session_state.software if 'software' in st.session_state else Path(dockm8_path / 'software')
	# Check if p2rank executable is available
	if not os.path.exists(software / "p2rank" / "prank"):
		p2rank_path = download_p2rank(software)
	else:
		pass
	try:
		st.write("Finding pockets using p2rank...")
		selected_pocket = select_pocket_from_local_protein(protein_input, p2rank_home=str(software / "p2rank"))
		if selected_pocket:
			st.write(
				f"Selected pocket coordinates: {selected_pocket['center'][0]}, {selected_pocket['center'][1]}, {selected_pocket['center'][2]}"
			)
	except Exception as e:
		st.error(f"Error in finding pocket: {str(e)}")
		st.error(f"Error type: {type(e).__name__}")
		import traceback
		st.error(f"Traceback: {traceback.format_exc()}")

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

elif pocket_mode == "Manual" and st.session_state["mode"] != "Single":
	st.error(
		"Manual pocket definition does not currently work in ensemble mode, please change the pocket definition mode")

else:
	st.warning("Please select a valid pocket detection mode.")

## Pocket Finding

## Visualisation

if pocket_mode == 'Reference' or pocket_mode == "RoG":
	# Visualize using docking mode
	st.header("Binding Site Visualization", divider='orange')
	st_molstar_docking(protein_input,
						reference_files[0],
						key="docking_vis",
						options={"defaultPolymerReprType": "cartoon"},
						height=900)

elif pocket_mode == "Dogsitescorer":
	st.header("Protein Visualization", divider='orange')
	st_molstar(protein_input, key="dogsite_vis", height=900)

elif pocket_mode == "Manual":
	st.header("Protein Visualization with Manual Pocket")
	st_molstar(protein_input, key="manual_vis", height=900)
	st.info(f"Manual pocket defined as: {manual_pocket}")
