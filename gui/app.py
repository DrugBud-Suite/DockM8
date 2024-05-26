import streamlit as st

from pages import page1_welcome
from pages import page2_library_analysis
from pages import page3_library_preparation
from pages import page4_protein_fetching_and_analysis
from pages import page5_protein_preparation
from pages import page6_binding_site_detection
from pages import page7_docking
from pages import page8_docking_postprocessing
from pages import page9_pose_selection
from pages import page10_rescoring
from pages import page11_consensus
from pages import page12_dockm8_report



st.sidebar.image(image="./media/DockM8_white_horizontal.png", width=200)
st.sidebar.subheader("Open-source consensus docking for everyone")

PAGES = {
	"Welcome": page1_welcome,
	"Library Analysis and Filtering": page2_library_analysis,
	"Library Preparation": page3_library_preparation,
	"Protein Fetching and Analysis": page4_protein_fetching_and_analysis,
	"Protein Preparation": page5_protein_preparation,
	"Binding Site Detection": page6_binding_site_detection,
	"Docking": page7_docking,
	"Docking Postprocessing": page8_docking_postprocessing,
	"Pose Selection": page9_pose_selection,
	"Rescoring": page10_rescoring,
	"Consensus": page11_consensus,
	"DockM8 Report": page12_dockm8_report}

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to: ", list(PAGES.keys()))

st.sidebar.link_button("Github", url="https://github.com/DrugBud-Suite/DockM8", use_container_width=True)
st.sidebar.link_button("Visit Website", url="https://drugbud-suite.github.io/dockm8-web/", use_container_width=True)
st.sidebar.link_button("Publication", url="https://doi.org/your-doi", use_container_width=True)
st.sidebar.link_button("Zenodo repository", url="https://doi.org/your-doi", use_container_width=True)

page = PAGES[selection]
page.app()
