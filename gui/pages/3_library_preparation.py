import sys
from pathlib import Path
import os
import pandas as pd
import streamlit as st
from rdkit.Chem import PandasTools
import traceback

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

from scripts.library_preparation.standardisation.standardise import standardize_library
from scripts.library_preparation.protonation.protgen_GypsumDL import protonate_GypsumDL
from scripts.library_preparation.conformer_generation.confgen_GypsumDL import generate_conformers_GypsumDL
from scripts.library_preparation.conformer_generation.confgen_RDKit import generate_conformers_RDKit
from gui.utils import save_dataframe_to_sdf
from gui.menu import menu, PAGES

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

if 'library_to_pass_to_preparation' not in st.session_state:
	st.session_state.library_to_pass_to_preparation = None

menu()

dockm8_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None).parent

st.title("Library Preparation", anchor='center')

st.subheader("Ligand library", divider="orange")
if st.session_state.library_to_pass_to_preparation is None:
	library_to_prepare_input = st.text_input(label="Enter the path to the ligand library file (.sdf format)",
												value=str(dockm8_path / "tests" / "test_files" / "library.sdf"),
												help="Choose a ligand library file (.sdf format)")
	# Button to load the library
	if st.button('Load Library'):
		if Path(library_to_prepare_input).is_file():
			st.session_state.library_to_pass_to_preparation = PandasTools.LoadSDF(library_to_prepare_input,
																					smilesName='SMILES',
																					molColName='Molecule',
																					idName='ID')
			st.write(f'Ligand library loaded with {len(st.session_state.library_to_pass_to_preparation)} compounds.')
		else:
			st.error('File does not exist.')
else:
	st.info("A library is already loaded.")

# Library Standardization
st.subheader("Library Standardization", divider="orange")

if 'library_to_pass_to_preparation' in st.session_state and st.session_state.library_to_pass_to_preparation is not None:
	col1, col2, col3 = st.columns(3)
	with col1:
		standardize_ids = st.toggle("Standardize IDs", value=True, help="Enable ID standardization")
	with col2:
		standardize_tautomers = st.toggle("Standardize Tautomers", value=True, help="Enable tautomer standardization")
	with col3:
		remove_salts = st.toggle("Remove Salts", value=True, help="Enable salt removal")

	# Add a button to run the standardization
	if st.button("Run Standardization"):
		try:
			standardized_df = standardize_library(
				df=st.session_state.library_to_pass_to_preparation,
				remove_salts=remove_salts,
				standardize_tautomers=standardize_tautomers,
				standardize_ids_flag=standardize_ids,
				n_cpus=st.session_state.n_cpus if 'n_cpus' in st.session_state else int(os.cpu_count() * 0.9))
			st.session_state.standardized_library = standardized_df
			st.success("Standardization completed successfully!")
			save_path = st.text_input('Enter path to save standardised library:',
										value=str(dockm8_path / 'tests' / 'test_files' / 'library_standardised.sdf'))
			if st.button('Save standardised library to SDF'):
				save_dataframe_to_sdf(st.session_state.standardized_library, save_path)
				st.success(f'Saved library to {save_path}')
		except Exception as e:
			st.error(f"An error occurred during standardization: {str(e)}")
			st.text(traceback.format_exc())
else:
	st.info("Please load a library first.")

# Ligand protonation
st.subheader("Ligand Protonation", divider="orange")

if 'standardized_library' in st.session_state:
	col1, col2, col3 = st.columns(3)
	with col1:
		min_ph = col1.number_input("Minimum pH for protonation", min_value=0.0, max_value=14.0, value=6.4, step=0.5)
	with col2:
		max_ph = col2.number_input("Maximum pH for protonation", min_value=0.0, max_value=14.0, value=8.4, step=0.5)
	with col3:
		pka_precision = col3.number_input("Precision of pH", min_value=0.0, max_value=2.0, value=1.0, step=0.5)

	# Add a button to run the standardization
	if st.button("Run Protonation"):
		try:
			protonated_df = protonate_GypsumDL(
				df=st.session_state.standardized_library,
				software=st.session_state.software if 'software' in st.session_state else Path(dockm8_path /
																								'software'),
				n_cpus=st.session_state.n_cpus if 'n_cpus' in st.session_state else int(os.cpu_count() * 0.9),
				min_ph=min_ph,
				max_ph=max_ph,
				pka_precision=pka_precision)

			st.session_state.protonated_library = protonated_df
			st.success("Protonation completed successfully!")
			save_path = st.text_input('Enter path to save protonated library:',
										value=str(dockm8_path / 'tests' / 'test_files' / 'library_protonated.sdf'))
			if st.button('Save protonated library to SDF'):
				save_dataframe_to_sdf(st.session_state.standardized_library, save_path)
				st.success(f'Saved library to {save_path}')
		except Exception as e:
			st.error(f"An error occurred during protonation: {str(e)}")
			st.text(traceback.format_exc())
else:
	st.info("Please standardise a library first.")

# Ligand conformers
st.subheader("Ligand Conformers", divider="orange")

if 'protonated_library' in st.session_state:
	library_to_generate_conformers = st.session_state.protonated_library
elif 'standardized_library' in st.session_state:
	library_to_generate_conformers = st.session_state.standardized_library
else:
	library_to_generate_conformers = st.session_state.library_to_pass_to_preparation

if library_to_generate_conformers is not None:
	conformer_method = st.selectbox(label="How should the conformers be generated?",
									options=["MMFF", "UFF", "GypsumDL"],
									index=1,
									help="MMFF: DockM8 will use MMFF forcefield to prepare the ligand 3D conformers. " +
									"UFF: DockM8 will use UFF forcefield to prepare the ligand 3D conformers. " +
									"GypsumDL: DockM8 will use Gypsum-DL to prepare the ligand 3D conformers.")
	# Add a button to run the standardization
	if st.button("Run Conformer Generation"):
		try:
			if conformer_method == 'MMFF':
				conformer_df = generate_conformers_RDKit(
					library_to_generate_conformers,
					n_cpus=st.session_state.n_cpus if 'n_cpus' in st.session_state else int(os.cpu_count() * 0.9),
					forcefield='MMFF')
			elif conformer_method == 'UFF':
				conformer_df = generate_conformers_RDKit(
					library_to_generate_conformers,
					n_cpus=st.session_state.n_cpus if 'n_cpus' in st.session_state else int(os.cpu_count() * 0.9),
					forcefield='UFF')
			elif conformer_method == 'GypsumDL':
				conformer_df = generate_conformers_GypsumDL(
					library_to_generate_conformers,
					software=st.session_state.software if 'software' in st.session_state else Path(dockm8_path /
																									'software'),
					n_cpus=st.session_state.n_cpus if 'n_cpus' in st.session_state else int(os.cpu_count() * 0.9))

			st.session_state.conformer_library = conformer_df
			st.success("Conformer generation completed successfully!")
			save_path = st.text_input('Enter path to save conformer library:',
										value=str(dockm8_path / 'tests' / 'test_files' / 'library_conformers.sdf'))
			if st.button('Save protonated library to SDF'):
				save_dataframe_to_sdf(st.session_state.conformer_library, save_path)
				st.success(f'Saved library to {save_path}')
		except Exception as e:
			st.error(f"An error occurred during conformer generation: {str(e)}")
			st.text(traceback.format_exc())
else:
	st.info("Please load a library first.")

if st.button('Proceed to Protein Preparation'):
	st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[3]))
