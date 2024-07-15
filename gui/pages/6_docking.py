import os
import sys
import traceback
from pathlib import Path
import pandas as pd
import streamlit as st

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import PAGES, menu
from scripts.docking.docking import DOCKING_PROGRAMS, concat_all_poses
from scripts.docking.docking_function import DockingFunction

menu()

st.title("Docking", anchor='center')

# Check for prepared docking library
if 'library_to_dock' not in st.session_state:
	default_path_library = Path(
		st.session_state.w_dir
	) / "prepared_library.sdf" if 'w_dir' in st.session_state else dockm8_path / "tests" / "test_files" / "prepared_library.sdf"
	library_to_dock_input = st.text_input(label="Enter the path to the ligand library file (.sdf format)",
				value=default_path_library,
				help="Choose a ligand library file (.sdf format)")
	if not Path(library_to_dock_input).is_file():
		st.error("File does not exist.")
	else:
		st.session_state.library_to_dock = library_to_dock_input
		st.success(f"Library loaded: {library_to_dock_input}")

# Check for prepared protein file
if 'prepared_protein_path' not in st.session_state:
	default_path_protein = Path(
		st.session_state.w_dir
	) / "prepared_protein.pdb" if 'w_dir' in st.session_state else dockm8_path / "tests" / "test_files" / "prepared_protein.pdb"
	st.warning("Prepared Protein File is missing.")
	protein_path = st.text_input("Enter the path to the prepared protein file (.pdb):",
			value=default_path_protein,
			help="Enter the complete file path to your prepared protein file.")
	if not Path(protein_path).is_file():
		st.error("File does not exist.")
	else:
		st.session_state.prepared_protein_path = protein_path
		st.success(f"Protein file loaded: {protein_path}")

# Check for binding site definition
if 'binding_site' not in st.session_state:
	st.warning("Binding Site Definition is missing.")
	if st.button("Define Binding Site"):
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[4]))

if 'software' not in st.session_state:
	st.info("Software path not set. Using default path. If you want to change it you can do so on the 'Setup' page.")
	st.session_state.software = dockm8_path / "software"

if 'docking_programs' not in st.session_state:
	st.session_state.docking_programs = None

# Display information about the available components
st.subheader("Docking Inputs", divider="orange")

# Custom CSS for styling
st.markdown("""
<style>
.metric-container {
	display: flex;
	align-items: center;
	margin-bottom: 10px;
	margin-top: 0px;
}
.metric-label {
	font-size: 18px;
	font-weight: bold;
	width: 180px;  /* Increased width to accommodate longer labels */
	margin-right: 10px;
	white-space: nowrap;  /* Prevent line breaks within the label */
}
.metric-value {
	font-size: 18px;
}
.subheader {
	font-size: 20px;
	font-weight: bold;
	margin-top: 0px;
	margin-bottom: 10px;
}
</style>
""",
	unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
	st.markdown("<div class='subheader'>Library and Protein</div>", unsafe_allow_html=True)

	if 'library_to_dock' in st.session_state:
		st.markdown(f"""
		<div class="metric-container">
			<span class="metric-label">Compounds to Dock</span>
			<span class="metric-value">{len(st.session_state.library_to_dock)}</span>
		</div>
		""",
			unsafe_allow_html=True)
	else:
		st.markdown("""
		<div class="metric-container">
			<span class="metric-label">Compounds to Dock</span>
			<span class="metric-value">Not loaded</span>
		</div>
		""",
			unsafe_allow_html=True)

	protein_path = st.session_state.get('prepared_protein_path', 'Not loaded')
	st.markdown(f"""
	<div class="metric-container">
		<span class="metric-label">Protein File</span>
		<span class="metric-value">{Path(protein_path).name if protein_path != 'Not loaded' else protein_path}</span>
	</div>
	""",
		unsafe_allow_html=True)

with col2:
	if 'binding_site' in st.session_state:
		st.markdown("<div class='subheader'>Binding Site</div>", unsafe_allow_html=True)

		subcol1, subcol2 = st.columns(2)

		with subcol1:
			for i, coord in enumerate(['X', 'Y', 'Z']):
				st.markdown(f"""
				<div class="metric-container">
					<span class="metric-label">{coord} Center</span>
					<span class="metric-value">{st.session_state.binding_site['center'][i]:.2f} Å</span>
				</div>
				""",
					unsafe_allow_html=True)

		with subcol2:
			for i, dim in enumerate(['Width', 'Height', 'Depth']):
				st.markdown(f"""
				<div class="metric-container">
					<span class="metric-label">{dim}</span>
					<span class="metric-value">{st.session_state.binding_site['size'][i]:.2f} Å</span>
				</div>
				""",
					unsafe_allow_html=True)
	else:
		st.markdown("<div class='subheader'>Binding Site</div>", unsafe_allow_html=True)
		st.markdown("""
		<div class="metric-container">
			<span class="metric-label">Status</span>
			<span class="metric-value">Not defined</span>
		</div>
		""",
			unsafe_allow_html=True)

# Docking programs
st.subheader("Docking Programs", divider="orange")
docking_programs = st.multiselect(label="Choose the docking programs you want to use:",
			default=["GNINA"],
			options=DOCKING_PROGRAMS,
			help="Select one or more docking programs. Multiple selections are allowed.")
st.session_state.docking_programs = docking_programs

if "PLANTS" in docking_programs and not os.path.exists(f"{st.session_state['software']}/PLANTS"):
	st.warning(
		"PLANTS was not found in the software folder. Please visit http://www.tcd.uni-konstanz.de/research/plants.php to download it."
	)
if "PANTHER" in docking_programs and not os.path.exists(f"{st.session_state['software']}/shaep"):
	st.warning(
		"SHAEP executable was not found in the software folder. Please visit https://users.abo.fi/mivainio/shaep/download.php to download it."
	)
if "FABIND+" in docking_programs and 'binding_site' in st.session_state:
	st.warning(
		"FABIND+ is a blind docking algorithm and does not require a binding site. Any previously defined binding site will be ignored (only for FABIND+)."
	)

# Docking parameters
st.subheader("Docking Parameters", divider="orange")

col1, col2 = st.columns(2)

with col1:
	st.session_state.n_poses = st.slider(label="Number of Poses",
				min_value=1,
				max_value=100,
				step=5,
				value=10,
				help="Specify the number of poses to generate for each ligand.")

with col2:
	st.session_state.exhaustiveness = st.select_slider(
		label="Exhaustiveness",
		options=[1, 2, 4, 8, 16, 32, 64],
		value=8,
		help=
		"Set the exhaustiveness of the docking search. Higher values can significantly increase the runtime. Only applies to GNINA, SMINA, QVINA2, QVINAW and PSOVINA."
	)

st.subheader("Run Docking", divider="orange")


def determine_working_directory() -> Path:
	if 'w_dir' in st.session_state:
		all_poses_path = Path(st.session_state.w_dir) / "allposes.sdf"
		return all_poses_path
	elif isinstance(st.session_state.library_to_dock, Path):
		all_poses_path = st.session_state.library_to_dock.parent / "allposes.sdf"
		return all_poses_path
	elif isinstance(st.session_state.library_to_dock, pd.DataFrame):
		custom_dir = st.text_input("Enter a custom save location:")
		# If user enters a file path
		if custom_dir.endswith(".sdf"):
			all_poses_path = Path(custom_dir)
			all_poses_path.parent.mkdir(exist_ok=True, parents=True)
			return all_poses_path
		elif "." in custom_dir and custom_dir.split(".")[-1] != "sdf":
			st.error("Please enter a valid .sdf file path or a directory.")
		# If user enters a directory path
		else:
			all_poses_path = Path(custom_dir) / "allposes.sdf"
			all_poses_path.parent.mkdir(exist_ok=True, parents=True)
			return all_poses_path
	st.error(
		"Unable to determine working directory. Please set a working directory or use a file path for the library.")
	return None


def run_docking():
	all_poses = []
	for program in docking_programs:
		docking_class = DOCKING_PROGRAMS[program]
		docking_function: DockingFunction = docking_class(st.session_state.software)

		common_params = {
			"library": st.session_state.library_to_dock,
			"protein_file": st.session_state.prepared_protein_path,
			"pocket_definition": st.session_state.binding_site,
			"exhaustiveness": st.session_state.get('exhaustiveness', 8),
			"n_poses": st.session_state.get('n_poses', 10),
			"n_cpus": st.session_state.get('n_cpus', int(os.cpu_count() * 0.9)), }

		if st.session_state.save_docking_results:
			allposes_save_path = st.session_state.poses_for_postprocessing
			output_sdf = allposes_save_path.parent / f"{program.lower()}_poses.sdf"
			docking_function.dock(**common_params, output_sdf=output_sdf)
		else:
			results = docking_function.dock(**common_params)
			all_poses.append(results)

	if st.session_state.save_docking_results:
		allposes_save_path = st.session_state.poses_for_postprocessing
		concat_all_poses(allposes_save_path, docking_programs, common_params["n_cpus"])
		st.session_state.poses_for_postprocessing = allposes_save_path
		st.info(f"All poses have been combined and saved to: {allposes_save_path}")
	else:
		st.session_state.poses_for_postprocessing = pd.concat(all_poses, ignore_index=True)


# UI Layout
col1, col2 = st.columns(2)
st.session_state.save_docking_results = col2.toggle(label="Save All Docking Results to SDF file",
				value=True,
				key='save_docking_results_toggle')

if st.session_state.save_docking_results:
	# Determine and set working directory
	allposes_save_path = determine_working_directory()
	if allposes_save_path:
		st.session_state.poses_for_postprocessing = allposes_save_path
		col2.write(f'Poses will be saved to: **{allposes_save_path}**')

# Run Docking Button
if col1.button('Run Docking', key='run_docking_button', use_container_width=True):
	if not docking_programs:
		st.error("Please select at least one docking program.")
	else:
		try:
			run_docking()
			st.success("Docking completed successfully!")
		except Exception as e:
			st.error(f"An error occurred during docking: {str(e)}")
			st.error(traceback.format_exc())

# Proceed to Postprocessing Button
if st.button('Proceed to Docking Postprocessing', key='proceed_to_docking_postprocessing_button'):
	if 'poses_for_postprocessing' in st.session_state:
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[6]))
	else:
		st.warning("Please run docking before proceeding to postprocessing.")
