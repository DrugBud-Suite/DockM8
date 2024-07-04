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

menu()

st.title("Docking Postprocessing")

st.subheader("Pose Minimization", divider="orange")
minimize_poses = st.toggle(label="Minimize poses",
							value=False,
							help="Minimize H positions of poses after docking",
							key="minimize_poses")

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
if st.button("Run Docking Postprocessing"):
	with st.spinner("Running docking postprocessing..."):
		if 'w_dir' not in st.session_state or 'prepared_protein_path' not in st.session_state:
			st.error("Please complete the previous steps before running docking postprocessing.")
		else:
			input_sdf = st.session_state.w_dir / "allposes.sdf"
			output_path = st.session_state.w_dir / "postprocessed_poses.sdf"
			protein_file = st.session_state.prepared_protein_path
			n_cpus = st.session_state.n_cpus if 'n_cpus' in st.session_state else int(os.cpu_count() * 0.9)

			try:
				result = docking_postprocessing(input_sdf=input_sdf,
												output_path=output_path,
												protein_file=protein_file,
												minimize_poses=minimize_poses,
												bust_poses=bust_poses,
												strain_cutoff=strain_cutoff,
												clash_cutoff=clash_cutoff,
												classy_pose=classy_pose,
												classy_pose_model=classy_pose_model,
												n_cpus=n_cpus)
				st.success(f"Docking postprocessing completed successfully. Results saved to {result}")
			except Exception as e:
				st.error(f"An error occurred during docking postprocessing: {str(e)}")
				st.write(traceback.format_exc())
	if st.button('Proceed to Pose Selection'):
		st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[8]))
