import sys
from pathlib import Path

import streamlit as st
from rdkit.Chem import PandasTools
import plotly.express as px
from rdkit.Chem import AllChem
from umap import UMAP
import pandas as pd
import numpy as np
import molplotly

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import menu
from scripts.library_analysis.descriptor_calculation import calculate_properties
from scripts.library_analysis.undesirable_filtering import query_chemfh, filter_by_properties
from gui.utils import filter_dataframe, display_dataframe
from scripts.library_analysis.chemical_space import visualize_chemical_space
import threading

menu()

st.title("Library Analysis and Filtering")

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

# Setup the path input and initial configuration
docking_library = st.text_input(label=r"$\textsf{\large Enter the path to the ligand library file (.sdf format)}$",
								value=str(dockm8_path / "tests" / "test_files" / "library.sdf"),
								help="Choose a ligand library file (.sdf format)")

# Button to load the library
if st.button("Load Library"):
	if Path(docking_library).is_file():
		ligand_library = PandasTools.LoadSDF(docking_library, smilesName="SMILES", molColName="Molecule", idName="ID")
		st.session_state['ligand_library'] = ligand_library
		st.write(f"Ligand library loaded with {len(ligand_library)} compounds.")
	else:
		st.error("File does not exist.")

with st.expander(r"$\textsf{\Large Property Calculation and Filtering}$"):
	if 'ligand_library' in st.session_state:
		st.write("##### Select the properties you want to calculate:")
		col1, col2, col3, col4 = st.columns(4)
		properties = {
			'MW': col1.checkbox('Molecular Weight (MW)', value=True),
			'TPSA': col2.checkbox('Polar Surface Area (TPSA)', value=True),
			'HBA': col3.checkbox('H-Bond Acceptors (HBA)', value=True),
			'HBD': col4.checkbox('H-Bond Donors (HBD)', value=True),
			'Rotatable Bonds': col1.checkbox('Number of Rotatable Bonds', value=True),
			'QED': col2.checkbox('Estimate of Drug-likeness (QED)', value=True),
			'sp3 percentage': col3.checkbox('sp3 Percentage', value=True),
			'Ring Count': col4.checkbox('Ring Count', value=True)}

		if st.button("Calculate Properties"):
			st.session_state['calculated_ligand_library'] = calculate_properties(st.session_state['ligand_library'],
																					properties)

	# Use session state data for filtering
	if 'calculated_ligand_library' in st.session_state:
		filtered_dataframe = filter_dataframe(st.session_state['ligand_library'])
		st.write(f"Filtered library contains {len(filtered_dataframe)} compounds. ")
		display_dataframe(filtered_dataframe.drop(['Molecule', 'SMILES'], axis=1), 'center')
		st.session_state['filtered_ligand_library'] = filtered_dataframe

if 'ligand_library' in st.session_state:
	with st.expander(r"$\textsf{\Large PAINS and Undesirable Compound Filtering}$"):
		if 'filtered_ligand_library' in st.session_state:
			library_to_filter_pains = st.session_state['filtered_ligand_library']
		elif 'calculated_ligand_library' in st.session_state:
			library_to_filter_pains = st.session_state['calculated_ligand_library']
		elif 'ligand_library' in st.session_state:
			library_to_filter_pains = st.session_state['ligand_library']
		else:
			st.write("Please load a library first.")
		if st.button("Query the ChemFH server for indesirable compounds"):
			result = query_chemfh(library_to_filter_pains)
			st.dataframe(result)

if 'ligand_library' in st.session_state:
	with st.expander(r"$\textsf{\Large Chemical Space Visualisation}$"):
		if 'filtered_ligand_library' in st.session_state:
			library_to_visualise = st.session_state['filtered_ligand_library']
		elif 'calculated_ligand_library' in st.session_state:
			library_to_visualise = st.session_state['calculated_ligand_library']
		elif 'ligand_library' in st.session_state:
			library_to_visualise = st.session_state['ligand_library']
		else:
			st.write("Please load a library first.")
		st.write("##### Select the fingerprint and plot type you want to use:")
		method = st.selectbox('Select Dimensionality Reduction Method:', ['UMAP', 'T-SNE', 'PCA'])
		fingerprint = st.selectbox('Select Fingerprint Type:', ['ECFP4', 'FCFP4', 'MACCS', 'Torsion'])
		col1, col2 = st.columns(2)
		if col1.button('Visualise Chemical Space', use_container_width=True):
			st.write("Calculating fingerprints and performing dimensionality reduction...")
			fig = visualize_chemical_space(library_to_visualise, method, fingerprint)
			st.plotly_chart(fig, theme="streamlit")
		if col2.button('Prepare Visualisation with Molecules', use_container_width=True):
			st.write("Calculating fingerprints and performing dimensionality reduction...")
			fig = visualize_chemical_space(library_to_visualise, method, fingerprint)
			app = molplotly.add_molecules(fig=fig, df=library_to_visualise, smiles_col='SMILES', title_col='ID')

			# Start the server in a separate thread
			thread = threading.Thread(target=lambda: app.run_server(mode='inline', port=8700, height=3000))
			thread.start()

			# Give a brief delay to ensure the server starts before displaying the link
			st.write('Server is starting, please wait...')
			button_html = """
			<a href="http://127.0.0.1:8700/" target="_blank" style="text-decoration: none;">
				<button style="color: white; background-color: #FF871F; border: none; padding: 5px 20px; 
				text-align: center; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; 
				border-radius: 8px;">Open Visualisation</button>
			</a>
			"""
			st.markdown(button_html, unsafe_allow_html=True)
