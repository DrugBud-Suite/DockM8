import streamlit as st
from pathlib import Path
from pages import *
from menu import menu

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

menu()

def app():
	st.columns(3)[1].image(image="./media/DockM8_white_vertical.png")

	st.markdown("<h1 style='text-align: center;'>Welcome to DockM8!</h1>", unsafe_allow_html=True)

	st.markdown("<h2 style='text-align: center;'>Choose a mode:</h2>", unsafe_allow_html=True)
	col1, col2 = st.columns(2)
	with col1:
		st.button("**Guided Mode**", type="primary", use_container_width=True)
	with col2:
		if st.button("**Advanced mode**", type="primary", use_container_width=True):
			st.switch_page("pages/2_library_analysis.py")

app()