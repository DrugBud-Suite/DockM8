import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.consensus.score_manipulation import standardize_scores
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS


@pytest.fixture
def sample_df():
	return pd.DataFrame({
		'Pose ID':
			range(1, 11),
		'AAScore':
			np.random.uniform(RESCORING_FUNCTIONS['AAScore']['score_range'][1],
								RESCORING_FUNCTIONS['AAScore']['score_range'][0],
								10),
		'AD4':
			np.random.uniform(RESCORING_FUNCTIONS['AD4']['score_range'][1],
								RESCORING_FUNCTIONS['AD4']['score_range'][0],
								10),
		'CNN-Score':
			np.random.uniform(RESCORING_FUNCTIONS['CNN-Score']['score_range'][0],
								RESCORING_FUNCTIONS['CNN-Score']['score_range'][1],
								10),
		'KORP-PL':
			np.random.uniform(RESCORING_FUNCTIONS['KORP-PL']['score_range'][1],
								RESCORING_FUNCTIONS['KORP-PL']['score_range'][0],
								10)})


def test_min_max_standardization(sample_df):
	result = standardize_scores(sample_df.copy(), 'min_max')
	for col in ['AAScore', 'AD4', 'CNN-Score', 'KORP-PL']:
		ascending = True if RESCORING_FUNCTIONS[col]['best_value'] == 'min' else False
		assert sample_df.sort_values(col, ascending=ascending).iloc[0]['Pose ID'] == result.sort_values(
			col, ascending=False).iloc[0]['Pose ID']


def test_scaled_standardization(sample_df):
	result = standardize_scores(sample_df.copy(), 'scaled')
	for col in ['AAScore', 'AD4', 'CNN-Score', 'KORP-PL']:
		ascending = True if RESCORING_FUNCTIONS[col]['best_value'] == 'min' else False
		assert sample_df.sort_values(col, ascending=ascending).iloc[0]['Pose ID'] == result.sort_values(
			col, ascending=False).iloc[0]['Pose ID']


def test_percentiles_standardization(sample_df):
	result = standardize_scores(sample_df.copy(), 'percentiles')
	for col in ['AAScore', 'AD4', 'CNN-Score', 'KORP-PL']:
		ascending = True if RESCORING_FUNCTIONS[col]['best_value'] == 'min' else False
		assert sample_df.sort_values(col, ascending=ascending).iloc[0]['Pose ID'] == result.sort_values(
			col, ascending=False).iloc[0]['Pose ID']


def test_invalid_standardization_type(sample_df):
	with pytest.raises(ValueError):
		standardize_scores(sample_df.copy(), 'invalid_type')
