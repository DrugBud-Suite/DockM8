import streamlit as st
from pathlib import Path
import sys

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import menu, PAGES
from scripts.consensus.consensus import CONSENSUS_METHODS

menu()


def app():
	st.title("Consensus")

	consensus_method = st.selectbox(label="Choose which consensus algorithm to use: ",
			index=2,
			options=list(CONSENSUS_METHODS.keys()),
			help="The method to use for consensus.")


app()
