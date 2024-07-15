import os
import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

from gui.menu import PAGES, menu
from scripts.library_preparation.library_preparation import prepare_library

menu()

st.title("Library Preparation", anchor='center')

# Ligand library
st.subheader("Ligand library", divider="orange")
if 'library_to_prepare' not in st.session_state:
	library_to_prepare_input = st.text_input(label="Enter the path to the ligand library file (.sdf format)",
				value=str(dockm8_path / "tests" / "test_files" / "library.sdf"),
				help="Choose a ligand library file (.sdf format)")
	if st.button('Load Library', key='load_library_button'):
		st.write('Loading library...')
		if Path(library_to_prepare_input).is_file():
			st.session_state.library_to_prepare = PandasTools.LoadSDF(library_to_prepare_input,
							smilesName='SMILES',
							molColName='Molecule',
							idName='ID')
			st.success(f'Ligand library loaded with {len(st.session_state.library_to_prepare)} compounds.')
		else:
			st.error('File does not exist.')
else:
	st.info("A library is already loaded.")

# Library Standardization
st.subheader("Library Standardization", divider="orange")
standardize_ids = st.toggle("Standardize IDs", value=True, help="Enable ID standardization")
standardize_tautomers = st.toggle("Standardize Tautomers", value=True, help="Enable tautomer standardization")
remove_salts = st.toggle("Remove Salts", value=True, help="Enable salt removal")

# Ligand protonation
st.subheader("Ligand Protonation", divider="orange")
protonation_method = st.selectbox(label="Protonation method",
			options=["GypsumDL", "None"],
			index=0,
			help="Choose the protonation method")
if protonation_method == "GypsumDL":
	min_ph = st.number_input("Minimum pH for protonation", min_value=0.0, max_value=14.0, value=6.4, step=0.5)
	max_ph = st.number_input("Maximum pH for protonation", min_value=0.0, max_value=14.0, value=8.4, step=0.5)
	pka_precision = st.number_input("Precision of pH", min_value=0.0, max_value=2.0, value=1.0, step=0.5)

# Ligand conformers
st.subheader("Ligand Conformers", divider="orange")
conformer_method = st.selectbox(label="Conformer generation method",
		options=["MMFF", "UFF", "GypsumDL"],
		index=0,
		help="Choose the conformer generation method")

st.subheader("Run Library Preparation", divider="orange")


def determine_working_directory() -> Path:
	if 'w_dir' in st.session_state:
		prepared_library_save_path = Path(st.session_state.w_dir) / "prepared_library.sdf"
		return prepared_library_save_path
	elif isinstance(st.session_state.library_to_prepare, Path):
		prepared_library_save_path = Path(st.session_state.library_to_prepare).parent / "prepared_library.sdf"
		return prepared_library_save_path
	else:
		custom_dir = st.text_input("Enter a custom save location:")
		if custom_dir.endswith(".sdf"):
			prepared_library_save_path = Path(custom_dir)
		else:
			prepared_library_save_path = Path(custom_dir) / "prepared_library.sdf"
		prepared_library_save_path.parent.mkdir(exist_ok=True, parents=True)
		return prepared_library_save_path


def run_library_preparation():
	common_params = {
		"input_data": st.session_state.library_to_prepare,
		"protonation": protonation_method,
		"conformers": conformer_method,
		"software": st.session_state.get('software', dockm8_path / 'software'),
		"n_cpus": st.session_state.get('n_cpus', int(os.cpu_count() * 0.9)),
		"standardize_ids": standardize_ids,
		"standardize_tautomers": standardize_tautomers,
		"remove_salts": remove_salts, }
	if protonation_method == "GypsumDL":
		common_params.update({"min_ph": min_ph, "max_ph": max_ph, "pka_precision": pka_precision, })

	if st.session_state.save_preparation_results:
		output_sdf = st.session_state.prepared_library_path
		prepare_library(**common_params, output_sdf=output_sdf)
		st.session_state.prepared_library = output_sdf
	else:
		results = prepare_library(**common_params)
		st.session_state.prepared_library = results


# UI Layout
col1, col2 = st.columns(2)
st.session_state.save_preparation_results = col2.toggle(label="Save Prepared Library to SDF file",
				value=True,
				key='save_preparation_results_toggle')

if st.session_state.save_preparation_results:
	prepared_library_save_path = determine_working_directory()
	if prepared_library_save_path:
		st.session_state.prepared_library_path = prepared_library_save_path
		col2.write(f'Prepared library will be saved to: **{prepared_library_save_path}**')

if col1.button("Run Library Preparation", use_container_width=True):
	try:
		run_library_preparation()
		st.success("Library preparation completed successfully.")
	except Exception as e:
		st.error(f"An error occurred during library preparation: {str(e)}")
		st.error(traceback.format_exc())

if st.button('Proceed to Protein Preparation'):
	if 'prepared_library' in st.session_state:
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[3]))
	else:
		st.warning("Library preparation was skipped, proceeding with the original library")
		st.session_state.prepared_library = st.session_state.library_to_prepare
