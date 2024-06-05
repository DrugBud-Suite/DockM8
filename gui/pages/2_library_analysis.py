import sys
from pathlib import Path

import streamlit as st
from rdkit.Chem import PandasTools
import plotly.express as px
from rdkit.Chem import AllChem
from umap import UMAP
import pandas as pd
import numpy as np

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import menu
from scripts.library_analysis.descriptor_calculation import calculate_properties
from gui.utils import filter_dataframe
from scripts.library_analysis.chemical_space import visualize_chemical_space

menu()


def plot_histograms(df, properties):
	for prop, selected in properties.items():
		if selected:
			fig = px.histogram(df, x=prop, nbins=30, title=prop, color_discrete_sequence=['orange'])
			fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
			st.plotly_chart(fig)


st.title("Library Analysis and Filtering")

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

docking_library = st.text_input(label="Entre the path to the ligand library file (.sdf format)",
								value=str(dockm8_path / "tests" / "test_files" / "library.sdf"),
								help="Choose a ligand library file (.sdf format)")

st.subheader("Select the properties you want to calculate:")
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

if 'ligand_library' not in st.session_state or st.button("Load Library and Calculate Properties"):
    if Path(docking_library).is_file():
        ligand_library = PandasTools.LoadSDF(docking_library,
                                             smilesName="SMILES",
                                             molColName="Molecule",
                                             idName="ID")
        st.write(f"Ligand library loaded with {len(ligand_library)} compounds.")
        st.session_state['ligand_library'] = calculate_properties(ligand_library, properties)
    else:
        st.error("File does not exist.")

# Use session state data for filtering
if 'ligand_library' in st.session_state:
    filtered_df = filter_dataframe(st.session_state['ligand_library'])
    
    st.write(f"Filtered library contains {len(filtered_df)} compounds. ")
    st.dataframe(filtered_df)

# Streamlit UI components to capture user input
method = st.selectbox('Select Dimensionality Reduction Method:', ['UMAP', 'TSNE', 'PCA'])
fingerprint = st.selectbox('Select Fingerprint Type:', ['ECFP4', 'FCFP4', 'MACCS', 'Torsion'])

# Visualize chemical space
if st.button("Visualise Chemical Space"):
	if 'ligand_library' in st.session_state:
		st.write("Calculating fingerprints and performing dimensionality reduction...")
		fig = visualize_chemical_space(st.session_state['ligand_library'], method, fingerprint)
		st.plotly_chart(fig)
	else:
		st.write("Please load and calculate the library first.")