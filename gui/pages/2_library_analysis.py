import sys
from pathlib import Path

import streamlit as st
from rdkit.Chem import PandasTools
import plotly.express as px

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import menu
from scripts.library_analysis.descriptor_calculation import calculate_properties

menu()


def plot_histograms(df, properties):
	for prop, selected in properties.items():
		if selected:
			fig = px.histogram(df, x=prop, nbins=30, title=prop, color_discrete_sequence=['orange'])
			fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
			st.plotly_chart(fig)


def app():
	st.title("Library Analysis and Filtering")

	# Search for 'DockM8' in parent directories
	gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
	dockm8_path = gui_path.parent
	sys.path.append(str(dockm8_path))

	docking_library = st.text_input(label="Entre the path to the ligand library file (.sdf format)",
									value=str(dockm8_path / "tests" / "test_files" / "library.sdf"),
									help="Choose a ligand library file (.sdf format)")
	
	st.subheader("Select the properties you want to calculate:")
	col1, col2, col3, col4, col5 = st.columns(5)
	properties = {
		'MW': col1.checkbox('Molecular Weight (MW)', value=True),
		'TPSA': col2.checkbox('Polar Surface Area (TPSA)', value=True),
		'LogD': col3.checkbox('LogD', value=True),
		'HBA': col4.checkbox('Hydrogen Bond Acceptors (HBA)', value=True),
		'HBD': col5.checkbox('Hydrogen Bond Donors (HBD)', value=True),
		'Rotatable Bonds': col1.checkbox('Number of Rotatable Bonds', value=True),
		'QED': col2.checkbox('Quantitative Estimate of Drug-likeness (QED)', value=True),
		'sp3 percentage': col3.checkbox('sp3 Carbon Percentage', value=True),
		'Ring Count': col4.checkbox('Ring Count', value=True)}

	if st.button("Load Library and Calculate Properties"):
		if Path(docking_library).is_file():
			ligand_library = PandasTools.LoadSDF(docking_library,
													smilesName="SMILES",
													molColName="Molecule",
													idName="ID")
			st.success("Library loaded and properties are being calculated...")
			ligand_library = calculate_properties(ligand_library, properties)
			st.dataframe(ligand_library.head())

			# Allow filtering on histograms
			st.subheader("Adjust the ranges for dynamic filtering:")
			filters = {}
			for prop in properties:
				if properties[prop]:
					col1, col2 = st.columns(2)
					with col1:
						min_val = float(ligand_library[prop].min())
						max_val = float(ligand_library[prop].max())
						filters[prop] = st.slider(f'Filter {prop}',
													min_val,
													max_val, (min_val, max_val),
													key=f"{prop}_filter")
					ligand_library = ligand_library[(ligand_library[prop] >= filters[prop][0]) &
													(ligand_library[prop] <= filters[prop][1])]

			plot_histograms(ligand_library, properties)
		else:
			st.error(f"Invalid file path: {docking_library}")


app()
