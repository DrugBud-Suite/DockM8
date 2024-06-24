import streamlit as st

def menu():
	st.sidebar.image(image="./media/DockM8_white_horizontal.png", width=200)
	st.sidebar.subheader("Open-source consensus docking for everyone")
	st.sidebar.subheader("", divider="orange")
	st.sidebar.page_link("app.py", label="Home")
	st.sidebar.page_link(f"pages/{PAGES[0]}", label="Setup")
	st.sidebar.page_link(f"pages/{PAGES[1]}", label="Library Analysis and Filtering")
	st.sidebar.page_link(f"pages/{PAGES[2]}", label="Library Preparation")
	st.sidebar.page_link(f"pages/{PAGES[3]}", label="Protein Fetching and Preparation")
	st.sidebar.page_link(f"pages/{PAGES[4]}", label="Binding Site Detection")
	st.sidebar.page_link(f"pages/{PAGES[5]}", label="Docking")
	st.sidebar.page_link(f"pages/{PAGES[6]}", label="Docking Postprocessing")
	st.sidebar.page_link(f"pages/{PAGES[7]}", label="Pose Selection")
	st.sidebar.page_link(f"pages/{PAGES[8]}", label="Rescoring")
	st.sidebar.page_link(f"pages/{PAGES[9]}", label="Consensus")
	st.sidebar.page_link(f"pages/{PAGES[10]}", label="DockM8 Report")
	st.sidebar.subheader("", divider="orange")
	col1, col3 = st.sidebar.columns(2)
	col1.link_button("Github", url="https://github.com/DrugBud-Suite/DockM8")
	col3.link_button("Visit Website", url="https://drugbud-suite.github.io/dockm8-web/")
	col1.link_button("Publication", url="https://doi.org/your-doi")
	col3.link_button("Zenodo", url="https://doi.org/your-doi")
	return


PAGES = [
	'1_setup.py',
	'2_library_analysis.py',
	'3_library_preparation.py',
	'4_protein_fetching_and_preparation.py',
	'5_binding_site_detection.py',
	'6_docking.py',
	'7_docking_postprocessing.py',
	'8_pose_selection.py',
	'9_rescoring.py',
	'10_consensus.py',
	'11_dockm8_report.py']
