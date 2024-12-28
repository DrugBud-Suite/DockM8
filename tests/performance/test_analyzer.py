import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

# Import the code to test
from scripts.performance.analyzer import ConsensusAnalyzer, normalize_scores, run_consensus_analysis
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS

# Add dummy scoring functions for testing
RESCORING_FUNCTIONS["score_func1"] = {"best_value": "max"}
RESCORING_FUNCTIONS["score_func2"] = {"best_value": "max"}
RESCORING_FUNCTIONS["score_func3"] = {"best_value": "min"}  # Add a min case for testing

# Fixtures for test data
@pytest.fixture
def sample_scoring_data():
    """Create sample scoring data for testing."""
    return pd.DataFrame({
        'ID': ['comp1', 'comp2', 'comp3', 'comp4'],
        'score_func1': [1.0, 2.0, 3.0, 4.0],
        'score_func2': [0.5, 1.5, 2.5, 3.5],
        'score_func3': [10.0, 20.0, 30.0, 40.0]
    })

@pytest.fixture
def sample_activity_data():
    """Create sample activity data for testing."""
    return pd.DataFrame({
        'ID': ['comp1', 'comp2', 'comp3', 'comp4'],
        'Activity': [1, 0, 1, 0]
    })

@pytest.fixture
def sample_custom_scoring_data():
    """Create sample scoring data for testing with custom columns."""
    return pd.DataFrame({
        'CustomID': ['comp1', 'comp2', 'comp3', 'comp4'],
        'score_func1': [1.0, 2.0, 3.0, 4.0],
        'score_func2': [0.5, 1.5, 2.5, 3.5],
        'score_func3': [10.0, 20.0, 30.0, 40.0]
    })

@pytest.fixture
def sample_custom_activity_data():
    """Create sample activity data for testing with custom columns."""
    return pd.DataFrame({
        'CustomID': ['comp1', 'comp2', 'comp3', 'comp4'],
        'CustomActivity': [1, 0, 1, 0]
    })

@pytest.fixture
def sample_empty_scoring_data():
    """Create empty scoring data for testing."""
    return pd.DataFrame({
        'ID': [],
        'score_func1': [],
        'score_func2': [],
        'score_func3': []
    })

@pytest.fixture
def sample_empty_activity_data():
    """Create empty activity data for testing."""
    return pd.DataFrame({
        'ID': [],
        'Activity': []
    })

@pytest.fixture
def temp_csv_files(sample_scoring_data, sample_activity_data):
    """Create temporary CSV files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scoring_path = Path(tmpdir) / "scoring.csv"
        activity_path = Path(tmpdir) / "activity.csv"
        
        sample_scoring_data.to_csv(scoring_path, index=False)
        sample_activity_data.to_csv(activity_path, index=False)
        
        yield scoring_path, activity_path

@pytest.fixture
def temp_csv_files_custom(sample_custom_scoring_data, sample_custom_activity_data):
    """Create temporary CSV files for testing with custom columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scoring_path = Path(tmpdir) / "scoring.csv"
        activity_path = Path(tmpdir) / "activity.csv"
        
        sample_custom_scoring_data.to_csv(scoring_path, index=False)
        sample_custom_activity_data.to_csv(activity_path, index=False)
        
        yield scoring_path, activity_path

@pytest.fixture
def temp_csv_files_empty(sample_empty_scoring_data, sample_empty_activity_data):
    """Create temporary CSV files with empty data for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scoring_path = Path(tmpdir) / "scoring.csv"
        activity_path = Path(tmpdir) / "activity.csv"
        
        sample_empty_scoring_data.to_csv(scoring_path, index=False)
        sample_empty_activity_data.to_csv(activity_path, index=False)
        
        yield scoring_path, activity_path

# Test normalize_scores function
def test_normalize_scores():
    """Test score normalization for different cases."""
    # Test normalization for maximize case
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    scoring_info = {"best_value": "max"}
    normalized = normalize_scores(scores, scoring_info)
    assert np.allclose(normalized, [0.0, 0.333333, 0.666667, 1.0], rtol=1e-5)
    
    # Test normalization for minimize case
    scoring_info = {"best_value": "min"}
    normalized = normalize_scores(scores, scoring_info)
    assert np.allclose(normalized, [1.0, 0.666667, 0.333333, 0.0], rtol=1e-5)

    # Test with negative scores
    scores = np.array([-4.0, -2.0, 0.0, 2.0])
    scoring_info = {"best_value": "max"}
    normalized = normalize_scores(scores, scoring_info)
    assert np.allclose(normalized, [0.0, 0.333333, 0.666667, 1.0], rtol=1e-5)
    
    # Test edge case - all same values
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    normalized = normalize_scores(scores, scoring_info)
    assert np.allclose(normalized, [0.5, 0.5, 0.5, 0.5])

# Test ConsensusAnalyzer initialization
def test_consensus_analyzer_init(temp_csv_files):
    """Test basic initialization of ConsensusAnalyzer."""
    scoring_path, activity_path = temp_csv_files
    analyzer = ConsensusAnalyzer(
        scoring_data_path=scoring_path,
        activity_data_path=activity_path
    )
    assert analyzer is not None
    assert analyzer.id_column == "ID"
    assert analyzer.activity_column == "Activity"
    assert len(analyzer.thresholds) > 0
    assert analyzer.n_jobs is None
    assert analyzer.batch_size == 100

def test_consensus_analyzer_init_custom_params(temp_csv_files_custom):
    """Test initialization with custom parameters."""
    scoring_path, activity_path = temp_csv_files_custom
    analyzer = ConsensusAnalyzer(
        scoring_data_path=scoring_path,
        activity_data_path=activity_path,
        id_column="CustomID",
        activity_column="CustomActivity",
        thresholds=[1, 2, 3],
        n_jobs=4,
        batch_size=50
    )
    assert analyzer.id_column == "CustomID"
    assert analyzer.activity_column == "CustomActivity"
    assert analyzer.thresholds == [1, 2, 3]
    assert analyzer.n_jobs == 4
    assert analyzer.batch_size == 50

# Test data validation
def test_missing_columns(temp_csv_files):
    """Test error handling for missing columns."""
    scoring_path, activity_path = temp_csv_files
    bad_data = pd.DataFrame({'Wrong_Column': [1, 2, 3]})
    bad_path = Path(scoring_path).parent / "bad_data.csv"
    bad_data.to_csv(bad_path, index=False)
    
    with pytest.raises(ValueError, match="Column 'ID' not found in scoring data"):
        ConsensusAnalyzer(
            scoring_data_path=bad_path,
            activity_data_path=activity_path
        )
        
    with pytest.raises(ValueError, match="Column 'ID' not found in activity data"):
        ConsensusAnalyzer(
            scoring_data_path=scoring_path,
            activity_data_path=bad_path
        )

def test_invalid_activity_values(temp_csv_files):
    """Test error handling for invalid activity values."""
    scoring_path, _ = temp_csv_files
    invalid_activity = pd.DataFrame({
        'ID': ['comp1', 'comp2', 'comp3', 'comp4'],
        'Activity': [1, 2, 3, 4]  # Should only be 0 or 1
    })
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
        invalid_activity.to_csv(f, index=False)
        invalid_path = f.name
    
    with pytest.raises(ValueError, match="Activity values must be binary"):
        ConsensusAnalyzer(
            scoring_data_path=scoring_path,
            activity_data_path=invalid_path
        )
    
    os.unlink(invalid_path)

def test_missing_values(temp_csv_files, sample_scoring_data, sample_activity_data):
    """Test error handling for missing values."""
    scoring_path, activity_path = temp_csv_files
    
    # Test missing values in scoring data
    bad_scoring = sample_scoring_data.copy()
    bad_scoring.iloc[0, 1] = np.nan
    bad_path = Path(scoring_path).parent / "bad_scoring.csv"
    bad_scoring.to_csv(bad_path, index=False)
    
    with pytest.raises(ValueError, match="Scoring data contains missing values"):
        ConsensusAnalyzer(
            scoring_data_path=bad_path,
            activity_data_path=activity_path
        )
    
    # Test missing values in activity data
    bad_activity = sample_activity_data.copy()
    bad_activity.iloc[0, 1] = np.nan
    bad_path = Path(activity_path).parent / "bad_activity.csv"
    bad_activity.to_csv(bad_path, index=False)
    
    with pytest.raises(ValueError, match="Activity data contains missing values"):
        ConsensusAnalyzer(
            scoring_data_path=scoring_path,
            activity_data_path=bad_path
        )

# Test single function analysis
def test_process_single_function(temp_csv_files):
    """Test processing of individual scoring functions."""
    scoring_path, activity_path = temp_csv_files
    analyzer = ConsensusAnalyzer(
        scoring_data_path=scoring_path,
        activity_data_path=activity_path
    )
    result = analyzer._process_single_function("score_func1")
    
    assert isinstance(result, pd.DataFrame)
    assert "scoring_function" in result.columns
    assert "threshold" in result.columns
    assert "n_compounds" in result.columns
    required_metrics = [
        "pm", "ef", "ref", "roce", "ccr", "mcc", "ef_alt", "rdkit_ef",
        "ckc", "auc_roc", "aupr", "bedroc", "rie", "rdkit_auc"
    ]
    assert all(metric in result.columns for metric in required_metrics)
    assert len(result) == 3  # Default 3 thresholds

def test_analyze_with_single_functions(temp_csv_files):
    """Test full analysis including single functions."""
    scoring_path, activity_path = temp_csv_files
    results = run_consensus_analysis(
        scoring_data_path=scoring_path,
        activity_data_path=activity_path,
        thresholds=[1],
        n_jobs=1,
        include_single=True
    )
    
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    
    # Check single function results
    single_results = results[results["scoring_function"].notna()]
    assert len(single_results) > 0
    assert all(func in single_results["scoring_function"].unique()
              for func in ["score_func1", "score_func2", "score_func3"])
    
    # Check consensus results
    consensus_results = results[results["combination"].notna()]
    assert len(consensus_results) > 0
    assert "consensus_method" in consensus_results.columns

def test_analyze_without_single_functions(temp_csv_files):
    """Test full analysis excluding single functions."""
    scoring_path, activity_path = temp_csv_files
    results = run_consensus_analysis(
        scoring_data_path=scoring_path,
        activity_data_path=activity_path,
        thresholds=[1],
        n_jobs=1,
        include_single=False
    )
    
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert "scoring_function" not in results.columns
    assert all(col in results.columns for col in [
        "combination", "consensus_method", "threshold"
    ])

def test_process_combination(temp_csv_files):
    """Test processing of scoring function combinations."""
    scoring_path, activity_path = temp_csv_files
    analyzer = ConsensusAnalyzer(
        scoring_data_path=scoring_path,
        activity_data_path=activity_path
    )
    # Test with a manually created combination
    combination = ("score_func1", "score_func2")
    result = analyzer._process_combination(combination, ["ecr"])
    
    assert isinstance(result, pd.DataFrame)
    required_columns = {
        "combination", "threshold", "consensus_method", "n_compounds",
        "pm", "ef", "ref", "roce", "ccr", "mcc", "ef_alt", "rdkit_ef",
        "ckc", "auc_roc", "aupr", "bedroc", "rie", "rdkit_auc"
    }
    assert all(col in result.columns for col in required_columns)
    assert len(result) == 3  # Default 3 thresholds

# Test parallel processing
def test_parallel_processing(temp_csv_files):
    """Test consistency of results between serial and parallel processing."""
    scoring_path, activity_path = temp_csv_files
    
    # Compare serial and parallel results
    results_serial = run_consensus_analysis(
        scoring_data_path=scoring_path,
        activity_data_path=activity_path,
        thresholds=[1],
        n_jobs=1,
        include_single=True
    )
    
    results_parallel = run_consensus_analysis(
        scoring_data_path=scoring_path,
        activity_data_path=activity_path,
        thresholds=[1],
        n_jobs=2,include_single=True
    )
    
    # Sort results for comparison
    sort_cols = ['combination', 'scoring_function', 'threshold', 'consensus_method']
    sort_cols = [col for col in sort_cols if col in results_serial.columns]
    
    results_serial_sorted = results_serial.sort_values(
        by=sort_cols
    ).reset_index(drop=True)
    results_parallel_sorted = results_parallel.sort_values(
        by=sort_cols
    ).reset_index(drop=True)
    
    pd.testing.assert_frame_equal(results_serial_sorted, results_parallel_sorted)

def test_normalization_with_single_functions(temp_csv_files):
    """Test score normalization with both max and min scoring functions."""
    scoring_path, activity_path = temp_csv_files
    analyzer = ConsensusAnalyzer(
        scoring_data_path=scoring_path,
        activity_data_path=activity_path
    )
    
    # Verify normalization for both max and min cases
    assert "score_func1" in analyzer.normalized_data.columns  # max case
    assert "score_func3" in analyzer.normalized_data.columns  # min case
    
    # Check max case normalization
    func1_scores = analyzer.normalized_data["score_func1"].values
    assert np.all((func1_scores >= 0) & (func1_scores <= 1))
    assert np.isclose(func1_scores.max(), 1.0)
    assert np.isclose(func1_scores.min(), 0.0)
    
    # Check min case normalization
    func3_scores = analyzer.normalized_data["score_func3"].values
    assert np.all((func3_scores >= 0) & (func3_scores <= 1))
    assert np.isclose(func3_scores.max(), 1.0)
    assert np.isclose(func3_scores.min(), 0.0)

def test_error_handling_single_functions(temp_csv_files, sample_scoring_data):
    """Test error handling for single function analysis."""
    scoring_path, activity_path = temp_csv_files
    analyzer = ConsensusAnalyzer(
        scoring_data_path=scoring_path,
        activity_data_path=activity_path
    )
    
    # Test invalid function name
    result = analyzer._process_single_function("invalid_function")
    assert "error" in result.columns
    assert len(result) == 1
    
    # Test with missing data
    bad_scoring = sample_scoring_data.copy()
    bad_scoring.iloc[0, 1] = np.nan
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
        bad_scoring.to_csv(f, index=False)
        bad_path = f.name
    
    with pytest.raises(ValueError, match="Scoring data contains missing values"):
        run_consensus_analysis(
            scoring_data_path=bad_path,
            activity_data_path=activity_path,
            include_single=True
        )
    
    os.unlink(bad_path)

def test_empty_data_handling(temp_csv_files_empty):
    """Test handling of empty datasets."""
    scoring_path, activity_path = temp_csv_files_empty
    with pytest.raises(ValueError, match="Empty scoring or activity file"):
        run_consensus_analysis(
            scoring_data_path=scoring_path,
            activity_data_path=activity_path,
            include_single=True
        )


def test_no_common_ids(temp_csv_files, sample_activity_data):
    """Test error handling when there are no common IDs between datasets."""
    scoring_path, activity_path = temp_csv_files
    # Create scoring data with no common IDs
    bad_scoring = pd.DataFrame({
        'ID': ['comp5', 'comp6', 'comp7', 'comp8'],
        'score_func1': [1.0, 2.0, 3.0, 4.0],
        'score_func2': [0.5, 1.5, 2.5, 3.5],
        'score_func3': [10.0, 20.0, 30.0, 40.0]
    })
    bad_path = Path(scoring_path).parent / "bad_scoring.csv"
    bad_scoring.to_csv(bad_path, index=False)
    
    with pytest.raises(ValueError, match="No common IDs found between scoring and activity data"):
        ConsensusAnalyzer(
            scoring_data_path=bad_path,
            activity_data_path=activity_path
        )

def test_output_file_handling(temp_csv_files):
    """Test output file creation and content."""
    scoring_path, activity_path = temp_csv_files
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        output_path = tmp.name
        
    try:
        results = run_consensus_analysis(
            scoring_data_path=scoring_path,
            activity_data_path=activity_path,
            output_path=output_path,
            thresholds=[1],
            n_jobs=1,
            include_single=True
        )
        
        # Verify file was created and contains correct data
        assert os.path.exists(output_path)
        loaded_results = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(results, loaded_results)
        
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)

def test_comprehensive_analysis(temp_csv_files):
    """Test comprehensive analysis with all features enabled."""
    scoring_path, activity_path = temp_csv_files
    results = run_consensus_analysis(
        scoring_data_path=scoring_path,
        activity_data_path=activity_path,
        thresholds=[1, 2, 5],
        n_jobs=1,
        include_single=True
    )
    
    print(results)
    
    # Verify both single and consensus results
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    
    # Check single function results
    single_results = results[results["scoring_function"].notna()]
    assert len(single_results) == 3 * 3  # functions * thresholds
    
    # Check consensus results
    consensus_results = results[results["combination"].notna()]
    assert len(consensus_results) > 0
    
    # Verify all metrics are present
    required_metrics = [
        "pm", "ef", "ref", "roce", "ccr", "mcc", "ef_alt", "rdkit_ef",
        "ckc", "auc_roc", "aupr", "bedroc", "rie", "rdkit_auc"
    ]
    assert all(metric in results.columns for metric in required_metrics)
    
    # Verify threshold values
    assert set(results["threshold"].unique()) == {1.0, 2.0, 5.0}


