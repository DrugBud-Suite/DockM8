import os
from pathlib import Path
import sys

import pandas as pd
import pytest

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.consensus.consensus_methods.avg_ECR import avg_ECR
from scripts.consensus.consensus_methods.avg_R_ECR import avg_R_ECR
from scripts.consensus.consensus_methods.ECR_avg import ECR_avg
from scripts.consensus.consensus_methods.ECR_best import ECR_best
from scripts.consensus.consensus_methods.RbR_best import RbR_best
from scripts.consensus.consensus_methods.RbR_avg import RbR_avg
from scripts.consensus.consensus_methods.RbV_best import RbV_best
from scripts.consensus.consensus_methods.RbV_avg import RbV_avg
from scripts.consensus.consensus_methods.Zscore_avg import Zscore_avg
from scripts.consensus.consensus_methods.Zscore_best import Zscore_best
from scripts.consensus.score_manipulation import rank_scores, standardize_scores


@pytest.fixture
def test_data():
	# Create a sample dataframe
	df = pd.read_csv(str(dockm8_path / "tests" / "test_files" / "consensus" / "allposes_rescored.csv"))
	standardized_df = standardize_scores(df, 'min_max')
	standardized_df["ID"] = standardized_df["Pose ID"].str.split("_").str[0]
	ranked_df = rank_scores(standardized_df)
	ranked_df["ID"] = ranked_df["Pose ID"].str.split("_").str[0]
	selected_columns = [
		"KORP-PL",
		"CHEMPLP",
		"NNScore",
		"CNN-Score",
		"AD4",
		"Vinardo",
		"GNINA-Affinity",
		"LinF9",
		"CNN-Affinity",
		"ConvexPLR",
		"RTMScore"]
	return standardized_df, ranked_df, selected_columns


def test_ECR_best(test_data):
	standardized_df, ranked_df, selected_columns = test_data
	expected_output = pd.DataFrame({
		'ID': ['FCG16141527', 'FCG16600623', 'FCG17822054', 'FCG16425532', 'FCG17585042', 'FCG18066182', 'FCG1390566'],
		'ECR_best': [
			1.0,
			0.9947674498716333,
			0.24268212158630068,
			0.21145415124141181,
			0.08505480001916879,
			0.0071077883662702766,
			0.0]})
	output = ECR_best(ranked_df, selected_columns).reset_index(drop=True)
	pd.testing.assert_frame_equal(output, expected_output, check_index_type=False)


def test_ECR_avg(test_data):
	standardized_df, ranked_df, selected_columns = test_data
	expected_output = pd.DataFrame({
		'ID': ['FCG1390566', 'FCG16141527', 'FCG16425532', 'FCG16600623', 'FCG17585042', 'FCG17822054', 'FCG18066182'],
		'ECR_avg': [
			0.0,
			0.819143729462059,
			0.08293276350859748,
			1.0,
			0.10194836276412086,
			0.17387747217794491,
			0.010157824388389454]})
	output = ECR_avg(ranked_df, selected_columns).reset_index(drop=True)
	pd.testing.assert_frame_equal(output, expected_output, check_index_type=False)


def test_avg_ECR(test_data):
	standardized_df, ranked_df, selected_columns = test_data
	expected_output = pd.DataFrame({
		'ID': ['FCG1390566', 'FCG16141527', 'FCG16425532', 'FCG16600623', 'FCG17585042', 'FCG17822054', 'FCG18066182'],
		'avg_ECR': [
			2.424818421603799e-06,
			0.04843430298365792,
			0.0,
			1.0,
			0.00631222571619443,
			1.029015873468495e-05,
			7.598878966962663e-09]})
	output = avg_ECR(ranked_df, selected_columns).reset_index(drop=True)
	pd.testing.assert_frame_equal(output, expected_output, check_index_type=False)


def test_avg_R_ECR(test_data):
	standardized_df, ranked_df, selected_columns = test_data
	expected_output = pd.DataFrame({
		'ID': ['FCG1390566', 'FCG16141527', 'FCG16425532', 'FCG16600623', 'FCG17585042', 'FCG17822054', 'FCG18066182'],
		'avg_R_ECR': [
			0.2351831858175574,
			1.0,
			0.0,
			0.5392849712705954,
			0.462815721489136,
			0.22324562885339497,
			0.012046980673249483]})
	output = avg_R_ECR(ranked_df, selected_columns).reset_index(drop=True)
	pd.testing.assert_frame_equal(output, expected_output, check_index_type=False)


def test_RbR_best(test_data):
	standardized_df, ranked_df, selected_columns = test_data
	expected_output = pd.DataFrame({
		'ID': ['FCG16141527', 'FCG16600623', 'FCG17585042', 'FCG17822054', 'FCG18066182', 'FCG16425532', 'FCG1390566'],
		'RbR_best': [
			1.0,
			0.7071129707112971,
			0.6652719665271967,
			0.606694560669456,
			0.34728033472803355,
			0.3054393305439331,
			0.0]})
	output = RbR_best(ranked_df, selected_columns).reset_index(drop=True)
	pd.testing.assert_frame_equal(output, expected_output, check_index_type=False)


def test_RbR_avg(test_data):
	standardized_df, ranked_df, selected_columns = test_data
	expected_output = pd.DataFrame({
		'ID': ['FCG1390566', 'FCG16141527', 'FCG16425532', 'FCG16600623', 'FCG17585042', 'FCG17822054', 'FCG18066182'],
		'RbR_avg': [
			0.0,
			1.0,
			0.2708708708708708,
			0.8168168168168168,
			0.7154654654654654,
			0.4834834834834835,
			0.3378378378378378]})
	output = RbR_avg(ranked_df, selected_columns).reset_index(drop=True)
	pd.testing.assert_frame_equal(output, expected_output, check_index_type=False)


def test_RbV_best(test_data):
	standardized_df, ranked_df, selected_columns = test_data
	expected_output = pd.DataFrame({
		'ID': ['FCG16600623', 'FCG16141527', 'FCG16425532', 'FCG17822054', 'FCG1390566', 'FCG18066182', 'FCG17585042'],
		'RbV_best': [1.0, 0.6666666666666666, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0]})
	output = RbV_best(standardized_df, selected_columns).reset_index(drop=True)
	pd.testing.assert_frame_equal(output, expected_output, check_index_type=False)


def test_RbV_avg(test_data):
	standardized_df, ranked_df, selected_columns = test_data
	expected_output = pd.DataFrame({
		'ID': ['FCG1390566', 'FCG16141527', 'FCG16425532', 'FCG16600623', 'FCG17585042', 'FCG17822054', 'FCG18066182'],
		'RbV_avg': [0.0, 0.532, 0.08, 1.0, 0.0, 0.2, 0.0]})
	output = RbV_avg(standardized_df, selected_columns).reset_index(drop=True)
	pd.testing.assert_frame_equal(output, expected_output, check_index_type=False)


def test_Zscore_best(test_data):
	standardized_df, ranked_df, selected_columns = test_data
	expected_output = pd.DataFrame({
		'ID': ['FCG16141527', 'FCG16600623', 'FCG17585042', 'FCG17822054', 'FCG18066182', 'FCG16425532', 'FCG1390566'],
		'Zscore_best': [
			1.0,
			0.7471298675481715,
			0.6361858729635781,
			0.5930816739433111,
			0.41570210997493745,
			0.41146421597635524,
			0.0]})
	output = Zscore_best(standardized_df, selected_columns).reset_index(drop=True)
	pd.testing.assert_frame_equal(output, expected_output, check_index_type=False)


def test_Zscore_avg(test_data):
	standardized_df, ranked_df, selected_columns = test_data
	expected_output = pd.DataFrame({
		'ID': ['FCG1390566', 'FCG16141527', 'FCG16425532', 'FCG16600623', 'FCG17585042', 'FCG17822054', 'FCG18066182'],
		'Zscore_avg': [
			0.0,
			1.0,
			0.24061193824767538,
			0.7823425229046653,
			0.6540988002926241,
			0.4887180475916245,
			0.40016972853383725]})
	output = Zscore_avg(standardized_df, selected_columns).reset_index(drop=True)
	pd.testing.assert_frame_equal(output, expected_output, check_index_type=False)
