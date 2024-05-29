import streamlit as st
from pathlib import Path
import sys

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import menu

menu()

def app():
	st.title("Docking Postprocessing")

	minimize_poses = st.toggle(label="Minimize poses",
								value=False,
								help="Minimize H positions of poses after docking",
								key="minimize_poses")

	clash_cutoff_toggle = st.toggle(label="Remove poses with clashes",
									value=True,
									help="Remove poses with clashes",
									key="clash_cutoff_toggle")
	if clash_cutoff_toggle:
		clash_cutoff = st.number_input(
			label=
			"Remove poses with more than x clashes: (setting too low will remove to many poses, use default if unsure)",
			min_value=0,
			max_value=100,
			value=5,
			step=1)

	else:
		clash_cutoff = None

	strain_cutoff_toggle = st.toggle(label="Remove poses with high strain",
										value=True,
										help="Remove poses with high strain",
										key="strain_cutoff_toggle")
	if strain_cutoff_toggle:
		strain_cutoff = st.number_input(
			label=
			"Remove poses with higher than x strain energy (kcal/mol): (setting too low will remove to many poses, use default if unsure)",
			min_value=100,
			max_value=100000,
			value=5000,
			step=100)

	else:
		strain_cutoff = None

	bust_poses = st.toggle(
		label="Bust poses using PoseBusters",
		value=True,
		help=
		"Bust poses using PoseBusters : Will remove any poses with clashes, non-flat aromatic rings etc. WARNING may take a long time to run",
		key="bust_poses_toggle")

	classy_pose = st.toggle(label="Classify poses using Classy_Pose",
							value=False,
							help="Classify poses using Classy_Pose",
							key="classy_pose_toggle")
	if classy_pose:
		classy_pose_model = st.selectbox(label="Choose the Classy_Pose model to use",
											options=("SVM (from publication)", "LGBM (retrained model)"),
											index=0)

app()
