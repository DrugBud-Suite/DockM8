import os
import sys
import traceback
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

if 'binding_site' not in st.session_state:
	st.session_state.binding_site = None

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
	for file in reference_files:
		if not Path(file).is_file():
			st.error(f"Invalid file path: {file}")

	if st.button("Find Pocket"):
		with st.spinner("Finding pocket..."):
			try:
				pocket_definition = find_pocket_default(reference_files[0], Path(protein_input), pocket_radius)
				st.session_state.binding_site = pocket_definition
				st.success("Pocket found successfully!")
			except Exception as e:
				st.error(f"Error in finding pocket: {str(e)}")
				st.error(traceback.format_exc())

elif pocket_mode == "RoG":
	reference_files = st.text_input(
		label="File path(s) of one or more multiple reference ligand files (.sdf format), separated by commas",
		help="Choose one or multiple reference ligand files (.sdf format)",
		value=str(dockm8_path / "tests" / "test_files" / "1fvv_l.sdf"),
	)
	reference_files = [Path(file.strip()) for file in reference_files.split(",")]
	for file in reference_files:
		if not Path(file).is_file():
			st.error(f"Invalid file path: {file}")

	if st.button("Find Pocket"):
		with st.spinner("Finding pocket..."):
			try:
				pocket_definition = find_pocket_RoG(reference_files[0], Path(protein_input))
				st.session_state.binding_site = pocket_definition
				st.success("Pocket found successfully!")
			except Exception as e:
				st.error(f"Error in finding pocket: {str(e)}")
				st.error(traceback.format_exc())

elif pocket_mode == "Dogsitescorer":
	dogsitescorer_mode = st.selectbox(label="Choose which metric to select binding sites by:",
										options=["Volume", "Druggability_Score", "Surface", "Depth"],
										help="Choose the metric to select binding sites by.")
	if st.button("Find Pockets"):
		with st.spinner("Finding pocket..."):
			try:
				pocket_definition = find_pocket_dogsitescorer(Path(protein_input), method=dogsitescorer_mode)
				st.session_state.binding_site = pocket_definition
				st.success("Pocket found successfully!")
			except Exception as e:
				st.error(f"Error in finding pocket: {str(e)}")
				st.error(traceback.format_exc())

elif pocket_mode == "p2rank":
	pocket_radius = st.number_input("Binding Site Radius", min_value=0.0, value=10.0, step=0.1)
	software = st.session_state.software if 'software' in st.session_state else Path(dockm8_path / 'software')
	with st.spinner("Finding pocket..."):
		if not os.path.exists(software / "p2rank" / "prank"):
			p2rank_path = download_p2rank(software)
		try:
			st.write("Finding pockets using p2rank...")
			selected_pocket = select_pocket_from_local_protein(protein_input, p2rank_home=str(software / "p2rank"))
			if selected_pocket:
				# Convert center coordinates to floats
				center = [float(coord) for coord in selected_pocket['center']]
				st.session_state.binding_site = {'center': center, 'size': [float(pocket_radius) * 2] * 3}
				st.write(f"Selected pocket coordinates: {center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}")
		except Exception as e:
			st.error(f"Error in finding pocket: {str(e)}")
			st.error(traceback.format_exc())

elif pocket_mode == "Manual" and st.session_state.get("mode") == "Single":
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

	if st.button("Set Manual Pocket"):
		st.session_state.binding_site = {'center': [x_center, y_center, z_center], 'size': [x_size, y_size, z_size]}
		st.success("Manual pocket set successfully!")

elif pocket_mode == "Manual" and st.session_state.get("mode") != "Single":
	st.error(
		"Manual pocket definition does not currently work in ensemble mode, please change the pocket definition mode")

else:
	st.warning("Please select a valid pocket detection mode.")

if st.session_state.binding_site:
	st.header("Binding Site Information", divider='orange')

	# Custom CSS for styling
	st.markdown("""
    <style>
    .metric-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
		margin-top: 0px;
    }
    .metric-label {
        font-size: 18px;
        font-weight: bold;
        width: 30px;
        margin-right: 10px;
    }
    .metric-value {
        font-size: 18px;
    }
    .subheader {
        font-size: 20px;
        font-weight: bold;
        margin-top: 0px;
        margin-bottom: 10px;
    }
    </style>
    """,
		unsafe_allow_html=True)

	col1, col2 = st.columns(2)

	with col1:
		st.markdown("<div class='subheader'>Center Coordinates</div>", unsafe_allow_html=True)
		for i, coord in enumerate(['X', 'Y', 'Z']):
			st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">{coord}</span>
                <span class="metric-value">{st.session_state.binding_site['center'][i]:.2f} Å</span>
            </div>
            """,
				unsafe_allow_html=True)

	with col2:
		st.markdown("<div class='subheader'>Pocket Dimensions</div>", unsafe_allow_html=True)
		for i, dim in enumerate(['Width', 'Height', 'Depth']):
			st.markdown(f"""
            <div class="metric-container">
                <span class="metric-label">{dim[0]}</span>
                <span class="metric-value">{st.session_state.binding_site['size'][i]:.2f} Å</span>
            </div>
            """,
				unsafe_allow_html=True)

	if st.button('Proceed to Docking', key='proceed_to_docking_button'):
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[5]))

	# Visualization section remains the same
	if pocket_mode == 'Reference' or pocket_mode == 'RoG':
		st.header("Binding Site Visualization", divider='orange')
		st.write(
			"DockM8 uses Mol* to view protein structures, you can find the documentation here : https://molstar.org/viewer-docs/"
		)
		st_molstar_docking(protein_input,
				reference_files[0],
				key="docking_vis",
				options={"defaultPolymerReprType": "cartoon"},
				height=900)
	elif pocket_mode in ["Dogsitescorer", "p2rank", "Manual"]:
		st.header("Protein Visualization", divider='orange')
		st.write(
			"DockM8 uses Mol* to view protein structures, you can find the documentation here : https://molstar.org/viewer-docs/"
		)
		st_molstar(protein_input, key="protein_vis", height=900)
