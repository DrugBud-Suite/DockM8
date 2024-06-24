import os
import sys
from pathlib import Path

import molplotly
import streamlit as st
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
gui_path = next((p / 'gui' for p in Path(__file__).resolve().parents if (p / 'gui').is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title='DockM8', page_icon='./media/DockM8_logo.png', layout='wide')

import threading

from gui.menu import menu, PAGES
from gui.utils import display_dataframe, filter_dataframe, save_dataframe_to_sdf
from scripts.library_analysis.chemical_space import visualize_chemical_space
from scripts.library_analysis.descriptor_calculation import calculate_properties
from scripts.library_analysis.medchem_filters import MEDCHEM_RULES, apply_medchem_rules
from scripts.library_analysis.alerts_filters import ALERTS_RULES, apply_alerts_rules
from scripts.library_analysis.undesirable_filtering import query_chemfh

menu()

st.title('Library Analysis and Filtering', anchor='center')

# Search for 'DockM8' in parent directories
gui_path = next((p / 'gui' for p in Path(__file__).resolve().parents if (p / 'gui').is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

# Setup the path input and initial configuration
docking_library = st.text_input(label='Enter the path to the ligand library file (.sdf format)',
								value=str(dockm8_path / 'tests' / 'test_files' / 'library.sdf'),
								help='Choose a ligand library file (.sdf format)')

# Button to load the library
if st.button('Load Library'):
	if Path(docking_library).is_file():
		ligand_library = PandasTools.LoadSDF(docking_library, smilesName='SMILES', molColName='Molecule', idName='ID')
		st.session_state['ligand_library'] = ligand_library
		st.write(f'Ligand library loaded with {len(ligand_library)} compounds.')
	else:
		st.error('File does not exist.')

if 'ligand_library' in st.session_state:
	st.subheader('Property Calculation and Filtering', divider='orange')
	st.write('##### Select the properties you want to calculate:')
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

	if st.button('Calculate Properties'):
		st.session_state['calculated_ligand_library'] = calculate_properties(st.session_state['ligand_library'],
																				properties)
		st.session_state['expander_property'] = True

	if 'calculated_ligand_library' in st.session_state:
		filtered_dataframe = filter_dataframe(st.session_state['ligand_library'])
		st.write(f'Filtered library contains {len(filtered_dataframe)} compounds. ')
		display_dataframe(filtered_dataframe.drop(['Molecule', 'SMILES'], axis=1), 'center')
		st.session_state['filtered_ligand_library'] = filtered_dataframe
		save_path = st.text_input('Enter path to save calculated properties SDF:',
									value=str(dockm8_path / 'tests' / 'test_files' / 'library_calculated.sdf'))
		if st.button('Save Calculated Properties to SDF'):
			save_dataframe_to_sdf(st.session_state['calculated_properties_df'], save_path)
			st.success(f'Saved library to {save_path}')

if 'ligand_library' in st.session_state:
	st.subheader('Medicinal Chemistry Filters', divider='orange')
	st.write('##### Select the medicinal chemistry filters you want to apply:')
	if 'filtered_ligand_library' in st.session_state:
		library_to_filter_medchem = st.session_state['filtered_ligand_library']
	elif 'calculated_ligand_library' in st.session_state:
		library_to_filter_medchem = st.session_state['calculated_ligand_library']
	elif 'ligand_library' in st.session_state:
		library_to_filter_medchem = st.session_state['ligand_library']
	else:
		st.write('Please load a library first.')
	selected_rules = []
	cols = st.columns(4)
	col_index = 0

	for rule_key, rule_info in MEDCHEM_RULES.items():
		rule_label = rule_info['alias']
		rule_description = f'{rule_info["alias"]}: {rule_info["rules"]} - {rule_info["description"]}'

		with cols[col_index]:
			if st.toggle(rule_label, help=rule_description):
				selected_rules.append(rule_key)

		col_index = (col_index+1) % 4

	if st.button('Apply Medicinal Chemistry Filters'):
		medchem_filtered_df, num_filtered, num_remaining = apply_medchem_rules(library_to_filter_medchem, selected_rules, st.session_state['n_cpus'] if 'n_cpus' in st.session_state else int(os.cpu_count()*0.9))
		st.write(
			f'Filtering complete. {num_filtered} compounds filtered, {num_remaining} compounds remaining. Displaying results:'
		)
		display_dataframe(medchem_filtered_df, 'center')
		st.session_state['filtered_ligand_library'] = medchem_filtered_df
		if st.button('Save Filtered Library to SDF'):
			save_dataframe_to_sdf(medchem_filtered_df, save_path)
			st.success(f'Saved library to {save_path}')

if 'ligand_library' in st.session_state:
	st.subheader('PAINS and Undesirable Compound Filtering', divider='orange')
	st.write('##### Select the undesirables and structural alert filters you want to apply:')
	if 'filtered_ligand_library' in st.session_state:
		library_to_filter_pains = st.session_state['filtered_ligand_library']
	elif 'calculated_ligand_library' in st.session_state:
		library_to_filter_pains = st.session_state['calculated_ligand_library']
	elif 'ligand_library' in st.session_state:
		library_to_filter_pains = st.session_state['ligand_library']
	else:
		st.write('Please load a library first.')
	selected_rules = []
	cols = st.columns(4)
	col_index = 0

	for rule_key, rule_info in ALERTS_RULES.items():
		with cols[col_index]:
			if st.toggle(rule_key, help=rule_info):
				selected_rules.append(rule_key)

		col_index = (col_index+1) % 4

	if st.button('Apply Structural Alerts Filters'):
		alerts_filtered_df, num_filtered, num_remaining = apply_alerts_rules(st.session_state['ligand_library'], selected_rules, st.session_state['n_cpus'] if 'n_cpus' in st.session_state else int(os.cpu_count()*0.9))
		st.write(
			f'Filtering complete. {num_filtered} compounds filtered, {num_remaining} compounds remaining. Displaying results:'
		)
		display_dataframe(alerts_filtered_df, 'center')
		st.session_state['filtered_ligand_library'] = alerts_filtered_df
	if st.button('Save Filtered Library to SDF'):
		save_dataframe_to_sdf(alerts_filtered_df, save_path)
		st.success(f'Saved library to {save_path}')
	if st.button('Query the ChemFH server for indesirable compounds'):
		result = query_chemfh(library_to_filter_pains)
		st.dataframe(result)

if 'ligand_library' in st.session_state:
	st.subheader('Chemical Space Visualisation', divider='orange')
	if 'filtered_ligand_library' in st.session_state:
		library_to_visualise = st.session_state['filtered_ligand_library']
	elif 'calculated_ligand_library' in st.session_state:
		library_to_visualise = st.session_state['calculated_ligand_library']
	elif 'ligand_library' in st.session_state:
		library_to_visualise = st.session_state['ligand_library']
	else:
		st.write('Please load a library first.')
	st.write('##### Select the fingerprint and plot type you want to use:')
	method = st.selectbox('Select Dimensionality Reduction Method:', ['UMAP', 'T-SNE', 'PCA'])
	fingerprint = st.selectbox('Select Fingerprint Type:', ['ECFP4', 'FCFP4', 'MACCS', 'Torsion'])
	col1, col2 = st.columns(2)
	if col1.button('Visualise Chemical Space', use_container_width=True):
		st.write('Calculating fingerprints and performing dimensionality reduction...')
		fig, embedding_df = visualize_chemical_space(library_to_visualise, method, fingerprint)
		st.plotly_chart(fig, theme='streamlit')
	if col2.button('Prepare Visualisation with Molecules', use_container_width=True):
		st.write('Calculating fingerprints and performing dimensionality reduction...')
		fig, embedding_df = visualize_chemical_space(library_to_visualise, method, fingerprint)
		app = molplotly.add_molecules(fig=fig, df=library_to_visualise, smiles_col='SMILES', title_col='ID')

		# Start the server in a separate thread
		thread = threading.Thread(target=lambda: app.run_server(mode='inline', port=8700, height=3000))
		thread.start()

		# Give a brief delay to ensure the server starts before displaying the link
		st.write('Server is starting, please wait...')
		button_html = '''
		<a href='http://127.0.0.1:8700/' target='_blank' style='text-decoration: none;'>
			<button style='color: white; background-color: #FF871F; border: none; padding: 5px 20px; 
			text-align: center; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; 
			border-radius: 8px;'>Open Visualisation</button>
		</a>
		'''
		st.markdown(button_html, unsafe_allow_html=True)

if 'ligand_library' in st.session_state:
	if 'filtered_ligand_library' in st.session_state:
		library_to_pass_to_preparation = st.session_state['filtered_ligand_library']
	elif 'calculated_ligand_library' in st.session_state:
		library_to_pass_to_preparation = st.session_state['calculated_ligand_library']
	elif 'ligand_library' in st.session_state:
		library_to_pass_to_preparation = st.session_state['ligand_library']
	else:
		st.write('Please load a library first.')
	col1, col2 = st.columns(2)
	if 'filtered_ligand_library' in st.session_state:
		if st.button('Save Filtered Library to SDF'):
			save_dataframe_to_sdf(st.session_state['filtered_ligand_library'], save_path)
			st.success(f'Saved library to {save_path}')
	if st.button('Proceed to Library Preparation'):
		st.session_state['library_to_pass_to_preparation'] = library_to_pass_to_preparation
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[2]))
