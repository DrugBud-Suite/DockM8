import sys
from pathlib import Path

import pandas as pd
from rdkit.Chem import PandasTools
import pytest

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.library_analysis.alerts_filters import apply_alerts_rules


@pytest.fixture
def common_test_data():
	"""Set up common test data."""
	dockm8_path = next((p for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
	library = dockm8_path / "tests/test_files/library_preparation/library.sdf"
	return library


def test_apply_alerts_rules(common_test_data):
	library = common_test_data

	df = PandasTools.LoadSDF(str(library), molColName="Molecule", idName="ID")
	# Define the selected rules for testing
	selected_rules = ['Frequent-Hitter']

	# Call the function to apply alerts rules
	filtered_df, num_filtered, num_remaining = apply_alerts_rules(df, selected_rules, n_cpus=2)

	assert isinstance(filtered_df, pd.DataFrame)
	assert isinstance(num_filtered, int)
	assert isinstance(num_remaining, int)
	assert len(filtered_df) <= len(df)
	assert 'Molecule' in filtered_df.columns
	assert 'ID' in filtered_df.columns
