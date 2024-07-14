import os
import sys
import traceback
from pathlib import Path

import streamlit as st

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import PAGES, menu
from scripts.docking_postprocessing.docking_postprocessing import docking_postprocessing
from gui.utils import save_dataframe_to_sdf

menu()

st.title("Docking Postprocessing")

if 'w_dir' not in st.session_state and 'prepared_protein_path' not in st.session_state:
	st.error("Please supply a file containing docked poses and a prepared protein file.")
	sdf_path = st.text_input("Enter the path to the docked poses (.sdf):",
								help="Enter the complete file path to your docked poses.",
								key="sdf_path")
	if sdf_path and Path(sdf_path).is_file():
		st.session_state.poses_for_postprocessing = sdf_path
		st.success(f"Pose file loaded: {sdf_path}")
	protein_path = st.text_input("Enter the path to the prepared protein file (.pdb):",
									help="Enter the complete file path to your prepared protein file.",
									key="protein_path")
	if protein_path and Path(protein_path).is_file():
		st.session_state.prepared_protein_path = protein_path
		st.success(f"Protein file loaded: {protein_path}")
elif 'w_dir' in st.session_state and 'prepared_protein_path' not in st.session_state:
	sdf_path = Path(st.session_state.w_dir) / "docking/allposes.sdf"
	if not Path(sdf_path).is_file():
		st.error(f"Could not find {sdf_path} in the working directory. Ensure it is in the docking folder.")
	elif Path(sdf_path).is_file():
		st.session_state.poses_for_postprocessing = sdf_path
		st.success(f"Pose file loaded: {sdf_path}")
	protein_path = st.text_input("Enter the path to the prepared protein file (.pdb):",
									value=Path(st.session_state.w_dir) / "prepared_protein.pdb",
									help="Enter the complete file path to your prepared protein file.",
									key="protein_path_2")
	if protein_path and Path(protein_path).is_file():
		st.session_state.prepared_protein_path = protein_path
		st.success(f"Protein file loaded: {protein_path}")
elif 'poses_for_postprocessing' in st.session_state and 'prepared_protein_path' in st.session_state:
	pass
else:
	st.error("Please supply a file containing docked poses and a prepared protein file.")

st.subheader("Pose Minimization", divider="orange")
col1, col2 = st.columns(2)
minimize_poses = col1.toggle(label="Minimize docking poses",
								value=False,
								help="Minimize poses after docking",
								key="minimize_poses")
if minimize_poses:
	minimize_poses_config = {}
	minimize_poses_config['force_field'] = col1.selectbox(label="Choose the force field for minimization",
															options=("UFF", "MMFF94", "MMFF94s"),
															index=2)
	minimize_poses_config['n_steps'] = col1.number_input(label="Number of minimization steps",
															min_value=1,
															max_value=5000,
															value=1000,
															step=10,
															help="Number of minimization steps to perform")
	minimize_poses_config['distance_constraing'] = col1.number_input(
		label="Distance constraint (Å)",
		min_value=0.1,
		max_value=2.0,
		value=1.0,
		step=0.1,
		help="Distance constraint in Å for minimization")
minimize_poses_H = col2.toggle(label="Minimize Hydrogens of docking poses",
								value=False,
								help="Minimize H positions of poses after docking",
								key="minimize_poses_H")

st.subheader("Clash and Strain Filtering", divider="orange")
col1, col2 = st.columns(2)
clash_cutoff_toggle = col1.toggle(label="Remove poses with clashes",
									value=True,
									help="Remove poses with clashes",
									key="clash_cutoff_toggle")
if clash_cutoff_toggle:
	clash_cutoff = col1.number_input(label="Remove poses with more than x clashes:",
										min_value=0,
										max_value=100,
										value=5,
										step=1,
										help="Setting too low will remove too many poses, use default if unsure")
else:
	clash_cutoff = None

strain_cutoff_toggle = col2.toggle(label="Remove poses with high strain",
									value=True,
									help="Remove poses with high strain",
									key="strain_cutoff_toggle")
if strain_cutoff_toggle:
	strain_cutoff = col2.number_input(label="Remove poses with higher than x strain energy (kcal/mol):",
										min_value=100,
										max_value=100000,
										value=5000,
										step=100,
										help="Setting too low will remove too many poses, use default if unsure")
else:
	strain_cutoff = None

st.subheader("PoseBusters Analysis", divider="orange")
bust_poses = st.toggle(
	label="Bust poses using PoseBusters",
	value=True,
	help=
	"Bust poses using PoseBusters: Will remove any poses with clashes, non-flat aromatic rings etc. WARNING: may take a long time to run",
	key="bust_poses_toggle")

st.subheader("Classy_Pose Classification", divider="orange")
classy_pose = st.toggle(label="Classify poses using Classy_Pose",
						value=False,
						help="Classify poses using Classy_Pose",
						key="classy_pose_toggle")
if classy_pose:
	classy_pose_model = st.selectbox(label="Choose the Classy_Pose model to use",
										options=("SVM (from publication)", "LGBM (retrained model)"),
										index=0)

st.subheader("Run Postprocessing", divider="orange")
st.session_state.save_postprocessed_poses = st.toggle(label="Save Postprocessed Poses to SDF file",
														value=True,
														key="save_postprocessed_poses_toggle")
if 'w_dir' in st.session_state:
	st.session_state.postprocessed_poses_save_path = Path(st.session_state.w_dir) / "postprocessed_poses.sdf"
elif isinstance(st.session_state.poses_for_postprocessing, Path):
	st.session_state.postprocessed_poses_save_path = st.session_state.poses_for_postprocessing.parent / "postprocessed_poses.sdf"
else:
	st.session_state.postprocessed_poses_save_path = st.text_input(
		"Enter the path to save the postprocessed poses (.sdf):",
		help="Enter the complete file path to save the postprocessed poses.",
		value="",
		key="save_path")
if st.button("Run Docking Postprocessing"):
	with st.spinner("Running docking postprocessing..."):
		try:
			result = docking_postprocessing(input_data=st.session_state.poses_for_postprocessing,
											protein_file=Path(st.session_state.prepared_protein_path),
											minimize_poses=minimize_poses,
											bust_poses=bust_poses,
											strain_cutoff=strain_cutoff,
											clash_cutoff=clash_cutoff,
											classy_pose=classy_pose,
											classy_pose_model=classy_pose_model if classy_pose else None,
											n_cpus=st.session_state.get('n_cpus', int(os.cpu_count() * 0.9)))
			st.success("Docking postprocessing completed successfully.")
			if st.session_state.save_postprocessed_poses:
				save_dataframe_to_sdf(result, st.session_state.postprocessed_poses_save_path)
				st.info(f"Postprocessed poses saved to {st.session_state.postprocessed_poses_save_path}")
		except FileNotFoundError as e:
			st.error(f"File not found: {e}")
		except ValueError as e:
			st.error(f"Invalid value: {e}")
		except Exception as e:
			st.error(f"An error occurred during docking postprocessing: {str(e)}")
			st.write(traceback.format_exc())
	if st.button('Proceed to Pose Selection'):
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[8]))
