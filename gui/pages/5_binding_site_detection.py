import sys
import traceback
from pathlib import Path

import streamlit as st
from streamlit_molstar import st_molstar
from streamlit_molstar.docking import st_molstar_docking
from streamlit_molstar.pocket import select_pocket_from_local_protein

gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import PAGES, menu
from scripts.pocket_finding.pocket_finder import PocketFinder, POCKET_DETECTION_OPTIONS

menu()
st.title("Binding Site Detection")

# Initialize session state variables
for key in ['binding_site', 'prepared_protein_path']:
    if key not in st.session_state:
        st.session_state[key] = None

# Set default protein path
protein_input = st.session_state.get('prepared_protein_path') or st.text_input(
    "Enter file path (.pdb):",
    value=str(Path(st.session_state.get('w_dir', dockm8_path / "tests" / "test_files")) / '1fvv_p.pdb'),
    help="Enter the complete file path to your protein data."
)

st.header("Binding Pocket definition", divider="orange")
pocket_mode = st.selectbox("Select a binding site detection method:", POCKET_DETECTION_OPTIONS,
                           help="Choose a method to define the binding pocket.")

# Initialize PocketFinder
pocket_finder = PocketFinder(software_path=st.session_state.get('software', Path(dockm8_path / 'software')))

def find_pocket_and_update_state(mode, **kwargs):
    try:
        pocket_definition = pocket_finder.find_pocket(mode=mode, receptor=Path(protein_input), **kwargs)
        st.session_state.binding_site = pocket_definition
        st.success("Pocket found successfully!")
    except Exception as e:
        st.error(f"Error in finding pocket: {str(e)}")
        st.error(traceback.format_exc())

if pocket_mode in ['Reference', 'RoG']:
    pocket_radius = st.number_input("Binding Site Radius", min_value=0.0, value=10.0, step=0.1) if pocket_mode == 'Reference' else None
    reference_files = [Path(file.strip()) for file in st.text_input(
        "File path(s) of reference ligand file(s) (.sdf format), separated by commas",
        value=str(dockm8_path / "tests" / "test_files" / "1fvv_l.sdf")).split(",")]
    
    if st.button("Find Pocket"):
        with st.spinner("Finding pocket..."):
            find_pocket_and_update_state(pocket_mode, ligand=reference_files[0], radius=pocket_radius)

elif pocket_mode == "Dogsitescorer":
    dogsitescorer_mode = st.selectbox("Choose which metric to select binding sites by:",
                                      ["Volume", "Druggability_Score", "Surface", "Depth"])
    if st.button("Find Pockets"):
        with st.spinner("Finding pocket..."):
            find_pocket_and_update_state(pocket_mode, dogsitescorer_method=dogsitescorer_mode)

elif pocket_mode == "p2rank":
    pocket_radius = st.number_input("Binding Site Radius", min_value=0.0, value=10.0, step=0.1)
    if st.button("Find Pockets"):
        with st.spinner("Finding pocket..."):
            try:
                selected_pocket = select_pocket_from_local_protein(protein_input, p2rank_home=str(st.session_state.get('software', Path(dockm8_path / 'software')) / "p2rank"))
                if selected_pocket:
                    st.session_state.binding_site = {
                        'center': [float(coord) for coord in selected_pocket['center']],
                        'size': [float(pocket_radius) * 2] * 3
                    }
                    st.write(f"Selected pocket coordinates: {', '.join([f'{coord:.2f}' for coord in st.session_state.binding_site['center']])} Å")
            except Exception as e:
                st.error(f"Error in finding pocket: {str(e)}")
                st.error(traceback.format_exc())

elif pocket_mode == "Manual" and st.session_state.get("mode") == "Single":
    col1, col2, col3 = st.columns(3)
    center = [col.number_input(f"{axis} Center", value=0.0) for col, axis in zip([col1, col2, col3], ['X', 'Y', 'Z'])]
    size = [col.number_input(f"{axis} Size", value=20.0) for col, axis in zip([col1, col2, col3], ['X', 'Y', 'Z'])]
    
    if st.button("Set Manual Pocket"):
        st.session_state.binding_site = {'center': center, 'size': size}
        st.success("Manual pocket set successfully!")

else:
    st.warning("Please select a valid pocket detection mode or ensure you're in Single mode for Manual detection.")

if st.session_state.binding_site:
    st.header("Binding Site Information", divider='orange')
    
    st.markdown("""
    <style>
    .metric-container { display: flex; align-items: center; margin-bottom: 10px; margin-top: 0px; }
    .metric-label { font-size: 18px; font-weight: bold; width: 30px; margin-right: 10px; }
    .metric-value { font-size: 18px; }
    .subheader { font-size: 20px; font-weight: bold; margin-top: 0px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for col, title, labels in zip([col1, col2], ['Center Coordinates', 'Pocket Dimensions'], [['X', 'Y', 'Z'], ['W', 'H', 'D']]):
        with col:
            st.markdown(f"<div class='subheader'>{title}</div>", unsafe_allow_html=True)
            for i, label in enumerate(labels):
                st.markdown(f"""
                <div class="metric-container">
                    <span class="metric-label">{label}</span>
                    <span class="metric-value">{st.session_state.binding_site['center' if title == 'Center Coordinates' else 'size'][i]:.2f} Å</span>
                </div>
                """, unsafe_allow_html=True)

    if st.button('Proceed to Docking', key='proceed_to_docking_button'):
        st.switch_page(str(dockm8_path / 'gui' / 'pages' / PAGES[5]))

    st.header(f"{'Binding Site' if pocket_mode in ['Reference', 'RoG'] else 'Protein'} Visualization", divider='orange')
    st.write("DockM8 uses Mol* to view protein structures. Documentation: https://molstar.org/viewer-docs/")
    
    if pocket_mode in ['Reference', 'RoG']:
        st_molstar_docking(protein_input, reference_files[0], key="docking_vis", options={"defaultPolymerReprType": "cartoon"}, height=900)
    else:
        st_molstar(protein_input, key="protein_vis", height=900)