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
	st.title("Protein Preparation")

	# Receptor(s)
	st.subheader("Receptor(s)", divider="orange")
	receptors = st.text_input(
		label="File path(s) of one or more multiple receptor files (.pdb format), separated by commas",
		help=
		"Choose one or multiple receptor files (.pdb format). Ensure there are no spaces in the file or directory names",
		value=str(dockm8_path / "tests" / "test_files" / "1fvv_p.pdb"),
		placeholder="Enter path(s) here")

	receptors = [Path(receptor.strip()) for receptor in receptors.split(",")]
	# Receptor files validation
	for file in receptors:
		if not Path(file).is_file():
			st.error(f"Invalid file path: {file}")

	# Prepare receptor
	st.subheader("Receptor Preparation", divider="orange")
	col1, col2, col3 = st.columns(3)
	select_best_chain = col1.toggle(label="AutoSelect best chain", key="select_best_chain", value=False)
	minimize_receptor = col2.toggle(label="Minimize receptor", key="minimize", value=True)
	if minimize_receptor:
		with_solvent = col2.toggle(label="Minimize with solvent", key="with_solvent", value=True)
	fix_nonstandard_residues = col3.toggle(label="Fix non standard residues", key="fix_nonstandard_residues", value=True)
	fix_missing_residues = col1.toggle(label="Fix missing residues", key="fix_missing_residues", value=True)
	remove_heteroatoms = col2.toggle(label="Remove ligands and heteroatoms", key="remove_heteroatoms", value=True)
	remove_water = col3.toggle(label="Remove water", key="remove_water", value=True)
	st.subheader("Receptor Protonation", divider="orange")
	protonation = st.toggle(label="Automatically protonate receptor using Protoss (untoggle to choose a specific pH)",
							value=True,
							help="Choose whether or not to use Protoss Web service to protonate the protein structure")

	if not protonation:
		add_hydrogens = st.number_input(label="Add hydrogens with PDB Fixer at pH",
										min_value=0.0,
										max_value=14.0,
										value=7.0)
	else:
		add_hydrogens = None

app()