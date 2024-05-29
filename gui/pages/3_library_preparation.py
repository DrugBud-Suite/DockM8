import streamlit as st
from pathlib import Path
import sys

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import menu

menu()

def app():

	dockm8_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None).parent

	st.title("Library Preparation")
	st.subheader("Ligand library", divider="orange")
	docking_library = st.text_input(label="Entre the path to the ligand library file (.sdf format)",
									value=str(dockm8_path / "tests" / "test_files" / "library.sdf"),
									help="Choose a ligand library file (.sdf format)")

	if not Path(docking_library).is_file():
		st.error(f"Invalid file path: {docking_library}")

	# Ligand protonation
	st.subheader("Ligand protonation", divider="orange")
	ligand_protonation = st.selectbox(label="How should the ligands be protonated?",
										options=("None", "GypsumDL"),
										index=1,
										help="None: No protonation " +
										"Gypsum-DL: DockM8 will use Gypsum-DL to protonate the ligands")

	# Ligand conformers
	st.subheader("Ligand conformers", divider="orange")
	ligand_conformers = st.selectbox(label="How should the conformers be generated?",
										options=["MMFF", "GypsumDL"],
										index=1,
										help="MMFF: DockM8 will use MMFF to prepare the ligand 3D conformers. " +
										"GypsumDL: DockM8 will use Gypsum-DL to prepare the ligand 3D conformers.")

	n_conformers = st.number_input("Number of conformers to generate.", min_value=1, max_value=100, step=1)

app()