import sys
import traceback
from pathlib import Path

import streamlit as st
from streamlit_molstar import st_molstar

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

from gui.menu import PAGES, menu
from scripts.protein_preparation.fetching.fetch_alphafold import fetch_alphafold_structure
from scripts.protein_preparation.fetching.fetch_pdb import fetch_pdb_structure
from scripts.protein_preparation.fixing.pdb_fixer import fix_pdb_file
from scripts.protein_preparation.minimization.minimization import minimize_receptor
from scripts.protein_preparation.protonation.protonate_protoss import protonate_protein_protoss
from scripts.protein_preparation.structure_assessment.edia import get_best_chain_edia

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

menu()

# Initialize session state variables
if 'add_missing_hydrogens' not in st.session_state:
	st.session_state.add_missing_hydrogens = True
if 'protonate' not in st.session_state:
	st.session_state.protonate = False


# Define callback functions
def toggle_add_hydrogens():
	st.session_state.add_missing_hydrogens = not st.session_state.add_missing_hydrogens
	st.session_state.protonate = not st.session_state.add_missing_hydrogens


def toggle_protonate():
	st.session_state.protonate = not st.session_state.protonate
	st.session_state.add_missing_hydrogens = not st.session_state.protonate


st.title("Protein Preparation", anchor='center', help="Main title for the protein preparation section.")

st.subheader("Input Protein", divider="orange", help="Section to input your protein data.")
col1, col2 = st.columns(2)
input_type = col1.radio("Select input type:", ("File Path", "PDB Code", "UniProt Code (AlphaFold structure)"),
						help="Select the type of input for the protein: a PDB code, UniProt code, or a file path.")

if input_type == "PDB Code":
	protein_input = col2.text_input("Enter PDB Code (4 characters):",
									help="Enter a valid PDB code consisting of 4 characters.")
elif input_type == "UniProt Code":
	protein_input = col2.text_input("Enter UniProt Code (6 characters):",
									help="Enter a valid UniProt code consisting of 6 characters.")
else:
	protein_input = col2.text_input("Enter file path (.pdb):",
									value=str(dockm8_path / "tests" / "test_files" / "1fvv_p.pdb"),
									help="Enter the complete file path to your protein data.")
if input_type == "File Path":
	if 'w_dir' in st.session_state:
		output_dir_value = str(st.session_state.w_dir)
	else:
		output_dir_value = str(dockm8_path / "tests" / "test_files")
else:
	output_dir_value = None

output_dir = col2.text_input("Output Directory:",
								value=output_dir_value,
								help="Specify the directory where the output will be saved.")

st.subheader("Preparation Options", divider="orange", help="Configure additional options for protein preparation.")

# Structure Selection
st.write("**Structure Selection**")
select_best_chain = st.toggle(
	"Select Best Chain (PDB only)",
	value=True,
	disabled=input_type != "PDB Code",
	help="Enable this to automatically select the best chain for PDB structures based on quality assessments.")

# Minimization
st.write("**Minimization**")
col1, col2 = st.columns(2)
with col1:
	minimize = st.toggle("Minimize Structure",
							value=False,
							help="Enable to minimize the structure to potentially improve quality.")
with col2:
	with_solvent = st.toggle("Include Solvent in Minimization",
								value=False,
								disabled=not minimize,
								help="Decide whether to include solvent molecules in the minimization process.")

# Structure Fixing
st.write("**Structure Fixing**")
fix_protein = st.toggle("Fix Protein",
						value=True,
						help="Enable to automatically fix issues in the protein structure, such as missing residues.")
if fix_protein:
	col1, col2 = st.columns(2)
	with col1:
		fix_nonstandard_residues = st.toggle("Fix Non-standard Residues",
												value=True,
												help="Enable to replace non-standard residues with standard ones.")
		fix_missing_residues = st.toggle("Fix Missing Residues",
											value=True,
											help="Enable to add missing residues to the structure.")
	with col2:
		remove_hetero = st.toggle("Remove Heteroatoms/Ligands",
									value=True,
									help="Enable to remove heteroatoms or ligands from the structure.")
		remove_water = st.toggle("Remove Water Molecules",
									value=True,
									help="Enable to remove water molecules from the structure.")

# Hydrogen Addition and Protonation
st.write("**Hydrogen Addition and Protonation**")
col1, col2 = st.columns(2)
with col1:
	add_missing_hydrogens = st.toggle("Add Missing Hydrogens",
										value=st.session_state.add_missing_hydrogens,
										on_change=toggle_add_hydrogens,
										help="Toggle to add missing hydrogen atoms to the protein structure.")
	if add_missing_hydrogens:
		add_missing_hydrogens_pH = st.number_input("pH for Adding Hydrogens",
													min_value=0.0,
													max_value=14.0,
													value=7.0,
													step=0.1,
													help="Set the pH value for adding hydrogens.")
with col2:
	protonate = st.toggle(
		"Protonate Protein with Protoss",
		value=st.session_state.protonate,
		on_change=toggle_protonate,
		help="Enable to use Protoss for protonating the protein, enhancing its chemical accuracy for simulations.")

if st.button("Prepare Protein"):
	with st.spinner("Preparing protein..."):
		try:
			# Input validation
			if input_type == "PDB Code" and (len(protein_input) != 4 or not protein_input.isalnum()):
				st.error("Invalid PDB code. It should be 4 alphanumeric characters.")
			elif input_type == "UniProt Code" and (len(protein_input) != 6 or not protein_input.isalnum()):
				st.error("Invalid UniProt code. It should be 6 alphanumeric characters.")
			elif input_type == "File Path" and not Path(protein_input).is_file():
				st.error("Invalid file path. The file does not exist.")
			else:
				# Prepare the protein
				output_path = Path(output_dir)
				output_path.mkdir(parents=True, exist_ok=True)

				if input_type == "PDB Code":
					if select_best_chain:
						input_protein = get_best_chain_edia(protein_input, output_path)
					else:
						input_protein = fetch_pdb_structure(protein_input, output_path)
				elif input_type == "UniProt Code":
					input_protein = fetch_alphafold_structure(protein_input, output_path)
				else:
					input_protein = Path(protein_input)

				# Minimize
				if minimize:
					minimized_protein = minimize_receptor(input_protein, solvent=with_solvent)
					current_protein = minimized_protein
				else:
					current_protein = input_protein

				# Fix protein
				if fix_protein:
					fixed_protein = fix_pdb_file(current_protein,
													output_path,
													fix_nonstandard_residues,
													fix_missing_residues,
													add_missing_hydrogens_pH if add_missing_hydrogens else None,
													remove_hetero,
													remove_water)
					current_protein = fixed_protein

				# Protonate
				if protonate:
					final_protein = protonate_protein_protoss(current_protein, output_path)
				else:
					final_protein = current_protein

				# Rename the final protein
				prepared_protein_path = output_path / "prepared_protein.pdb"
				final_protein.rename(prepared_protein_path)
				st.session_state.prepared_protein_path = prepared_protein_path
				st.success(f"Protein preparation completed. Output saved to: {prepared_protein_path}")
		except Exception as e:
			st.error(f"An error occurred during protein preparation: {str(e)}")
			st.error(traceback.format_exc())

if 'prepared_protein_path' in st.session_state:
	if st.button('Proceed to Binding Site Detection'):
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[4]))

# Add a new section for Protein Visualization
st.subheader("Protein Visualization",
				divider="orange",
				help="Visualize the protein structure using the Mol* web-based toolkit.")

# Check if the protein has been prepared and a path is available
if 'prepared_protein_path' in st.session_state:
	protein_file_path = str(st.session_state.prepared_protein_path)
	st.write(
		"DockM8 uses Mol* to view protein structures, you can find the documentation here : https://molstar.org/viewer-docs/"
	)
	if Path(protein_file_path).is_file():
		st_molstar(protein_file_path, height="900px", key='visualization_key')
	else:
		st.error("Protein file does not exist or path is incorrect.")
else:
	st.error("Protein has not been prepared yet. Please complete the preparation steps first.")
