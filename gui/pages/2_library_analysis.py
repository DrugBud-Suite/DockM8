import os
import sys
import traceback
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

from gui.menu import PAGES, menu
from gui.utils import filter_dataframe, save_dataframe_to_sdf
from scripts.utilities.utilities import parallel_SDF_loader
from scripts.library_analysis.alerts_filters import ALERTS_RULES, apply_alerts_rules
from scripts.library_analysis.chemical_space import visualize_chemical_space
from scripts.library_analysis.descriptor_calculation import calculate_properties
from scripts.library_analysis.medchem_filters import MEDCHEM_RULES, apply_medchem_rules

menu()

st.title('Library Analysis and Filtering', anchor='center')

# Setup the path input and initial configuration
input_library = st.text_input(label='Enter the path to the ligand library file (.sdf format)',
								value=str(dockm8_path / 'tests' / 'test_files' / 'library.sdf'),
								help='Choose a ligand library file (.sdf format)')

# Button to load the library
if st.button('Load Library'):
	with st.spinner('Loading library...'):
		if Path(input_library).is_file():
			n_cpus = st.session_state.get('n_cpus', int(os.cpu_count() * 0.9))
			ligand_library = parallel_SDF_loader(input_library,
													SMILES='SMILES',
													molColName='Molecule',
													idName='ID',
													n_cpus=n_cpus)
			st.session_state.ligand_library = ligand_library
			st.write(f'Ligand library loaded with {len(ligand_library)} compounds.')
		else:
			st.error('File does not exist.')

# Property Calculation and Filtering
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

	# Calculate properties
	if st.button('Calculate Properties'):
		with st.spinner('Calculating properties...'):
			try:
				st.session_state.calculated_ligand_library = calculate_properties(st.session_state.ligand_library,
																					properties)
				st.write(f"Properties calculated for {len(st.session_state.calculated_ligand_library)} compounds.")
				st.session_state.expander_property = True
			except Exception as e:
				st.error(f"An error occurred during property calculation: {str(e)}")
				st.error(traceback.format_exc())

	# Filter dataframe
	if 'calculated_ligand_library' in st.session_state:
		with st.spinner('Filtering dataframe...'):
			filtered_dataframe = filter_dataframe(st.session_state.calculated_ligand_library)
			st.write(f'Filtered library contains {len(filtered_dataframe)} compounds. ')
			st.session_state.filtered_ligand_library = filtered_dataframe

# Medicinal Chemistry Filters
if 'ligand_library' in st.session_state:
	st.subheader('Medicinal Chemistry Filters', divider='orange')
	st.write('##### Select the medicinal chemistry filters you want to apply:')
	if 'filtered_ligand_library' in st.session_state:
		library_to_filter_medchem = st.session_state.filtered_ligand_library
	elif 'calculated_ligand_library' in st.session_state:
		library_to_filter_medchem = st.session_state.calculated_ligand_library
	else:
		library_to_filter_medchem = st.session_state.ligand_library

	selected_medchem_rules = []
	cols = st.columns(4)
	col_index = 0

	# Display the MedChem rules as toggles
	for rule_key, rule_info in MEDCHEM_RULES.items():
		rule_label = rule_info['alias']
		rule_description = f'{rule_info["alias"]}: {rule_info["rules"]} - {rule_info["description"]}'
		with cols[col_index]:
			if st.toggle(rule_label, help=rule_description):
				selected_medchem_rules.append(rule_key)
		col_index = (col_index+1) % 4

	# Apply MedChem filters
	if st.button('Apply Medicinal Chemistry Filters'):
		with st.spinner('Applying medicinal chemistry filters...'):
			try:
				medchem_filtered_df, num_filtered, num_remaining = apply_medchem_rules(library_to_filter_medchem, selected_medchem_rules, st.session_state.n_cpus if 'n_cpus' in st.session_state else int(os.cpu_count()*0.9))
				st.write(f'Filtering complete. {num_filtered} compounds filtered, {num_remaining} compounds remaining.')
				st.session_state.filtered_ligand_library = medchem_filtered_df
			except Exception as e:
				st.error(f"An error occurred during filtering: {str(e)}")
				st.error(traceback.format_exc())

# PAINS and Undesirable Compound Filtering
if 'ligand_library' in st.session_state:
	st.subheader('PAINS and Undesirable Compound Filtering', divider='orange')
	st.write('##### Select the undesirables and structural alert filters you want to apply:')
	if 'filtered_ligand_library' in st.session_state:
		library_to_filter_pains = st.session_state.filtered_ligand_library
	elif 'calculated_ligand_library' in st.session_state:
		library_to_filter_pains = st.session_state.calculated_ligand_library
	else:
		library_to_filter_pains = st.session_state.ligand_library

	selected_alerts_rules = []
	cols = st.columns(4)
	col_index = 0

	# Display the PAINS and structural alerts as toggles
	for rule_key, rule_info in ALERTS_RULES.items():
		with cols[col_index]:
			if st.toggle(rule_key, help=rule_info):
				selected_alerts_rules.append(rule_key)
		col_index = (col_index+1) % 4

	# Apply PAINS and structural alerts filters
	if st.button('Apply Structural Alerts Filters'):
		with st.spinner('Applying structural alerts filters...'):
			try:
				alerts_filtered_df, num_filtered, num_remaining = apply_alerts_rules(library_to_filter_pains, selected_alerts_rules, st.session_state.n_cpus if 'n_cpus' in st.session_state else int(os.cpu_count()*0.9))
				st.write(f'Filtering complete. {num_filtered} compounds filtered, {num_remaining} compounds remaining.')
				st.session_state.filtered_ligand_library = alerts_filtered_df
			except Exception as e:
				st.error(f"An error occurred during filtering: {str(e)}")
				st.error(traceback.format_exc())

# Chemical Space Visualisation
if 'ligand_library' in st.session_state:
	st.subheader('Chemical Space Visualisation', divider='orange')
	if 'filtered_ligand_library' in st.session_state:
		library_to_visualise = st.session_state.filtered_ligand_library
	elif 'calculated_ligand_library' in st.session_state:
		library_to_visualise = st.session_state.calculated_ligand_library
	else:
		library_to_visualise = st.session_state.ligand_library

	st.write('##### Select the fingerprint and plot type you want to use:')
	method = st.selectbox('Select Dimensionality Reduction Method:', ['UMAP', 'T-SNE', 'PCA'])
	fingerprint = st.selectbox('Select Fingerprint Type:', ['ECFP4', 'FCFP4', 'MACCS', 'Torsion'])
	col1, col2 = st.columns(2)

	# Visualise chemical space (no structures)
	if col1.button('Visualise Chemical Space', use_container_width=True):
		with st.spinner('Calculating fingerprints and performing dimensionality reduction...'):
			try:
				fig, embedding_df = visualize_chemical_space(library_to_visualise, method, fingerprint)
				st.plotly_chart(fig, theme='streamlit')
			except Exception as e:
				st.error(f"An error occurred during visualisation: {str(e)}")
				st.error(traceback.format_exc())

	# Visualise chemical space (with structures)
	if col2.button('Prepare Visualisation with Molecules', use_container_width=True):
		with st.spinner('Calculating fingerprints and performing dimensionality reduction...'):
			try:
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
			except Exception as e:
				st.error(f"An error occurred during visualisation: {str(e)}")
				st.error(traceback.format_exc())

# Final Save and Proceed to Library Preparation
if 'ligand_library' in st.session_state:
	st.subheader('Save Final Library and Proceed', divider='orange')

	def determine_working_directory() -> Path:
		if 'w_dir' in st.session_state:
			final_library_save_path = Path(st.session_state.w_dir) / "filtered_library.sdf"
			return final_library_save_path
		elif 'ligand_library' in st.session_state:
			final_library_save_path = Path(input_library).parent / "filtered_library.sdf"
			return final_library_save_path
		else:
			custom_dir = st.text_input("Enter a custom save location:")
			if custom_dir.endswith(".sdf"):
				final_library_save_path = Path(custom_dir)
			else:
				final_library_save_path = Path(custom_dir) / "filtered_library.sdf"
			final_library_save_path.parent.mkdir(exist_ok=True, parents=True)
			return final_library_save_path

	def save_final_library():
		if 'filtered_ligand_library' in st.session_state:
			library_to_save = st.session_state.filtered_ligand_library
		elif 'calculated_ligand_library' in st.session_state:
			library_to_save = st.session_state.calculated_ligand_library
		else:
			library_to_save = st.session_state.ligand_library

		if st.session_state.save_final_library:
			output_sdf = st.session_state.final_library_path
			save_dataframe_to_sdf(library_to_save, output_sdf)
			st.success(f'Saved library to {output_sdf}')

		st.session_state.library_to_pass_to_preparation = library_to_save

	col1, col2 = st.columns(2)

	st.session_state.save_final_library = col2.toggle(label="Save Final Library to SDF file",
														value=True,
														key='save_final_library_toggle')

	if st.session_state.save_final_library:
		final_library_save_path = determine_working_directory()
		if final_library_save_path:
			st.session_state.final_library_path = final_library_save_path
			col2.write(f'Library will be saved to: **{final_library_save_path}**')

	if col1.button('Proceed to Library Preparation'):
		try:
			save_final_library()
			st.success("Final library saved successfully.")
			st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[2]))
		except Exception as e:
			st.error(f"An error occurred while saving the final library: {str(e)}")
			st.error(traceback.format_exc())
