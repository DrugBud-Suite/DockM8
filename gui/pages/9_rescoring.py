import streamlit as st
from pathlib import Path
import sys

# Search for 'DockM8' in parent directories
gui_path = next((p / "gui" for p in Path(__file__).resolve().parents if (p / "gui").is_dir()), None)
dockm8_path = gui_path.parent
sys.path.append(str(dockm8_path))

st.set_page_config(page_title="DockM8", page_icon="./media/DockM8_logo.png", layout="wide")

from gui.menu import menu, PAGES
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS

menu()

def app():
	st.title("Rescoring")

	st.subheader("Scoring functions", divider="orange")
	rescoring = st.multiselect(label="Choose the scoring functions you want to use",
			default=["CNN-Score", "KORP-PL"],
			options=list(RESCORING_FUNCTIONS.keys()),
			help="The method(s) to use for scoring. Multiple selection allowed")

	st.subheader("Score manipulation", divider="orange")
	st.toggle(label="Normalize scores", value=True, help="Normalize scores to a range of 0-1", key="normalize_scores")
	st.toggle(label="Normalize to MW", value=False, help="Scale the scores to molecular wwight of the compound", key="mw_scores")
app()
