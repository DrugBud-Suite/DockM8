import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

# Import the modules we need to test
import sys
from pathlib import Path

from scripts.consensus.consensus import apply_consensus_scoring
from scripts.performance.analyzer import run_consensus_analysis
from scripts.performance.calculate_metrics import calculate_metrics
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS
from scripts.consensus.consensus import _METHODS


def normalize_scores(scores: np.ndarray, best_value: str = "min") -> np.ndarray:
    """Normalize scores to [0,1] range where 1 is always best."""
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.full_like(scores, 0.5)
        
    normalized = (scores - min_score) / (max_score - min_score)
    if best_value == "min":
        normalized = 1 - normalized
    return normalized

def print_data_info(scores: np.ndarray, labels: np.ndarray, threshold: float):
    """Print detailed information about the data being processed."""
    print("\nData Overview:")
    print(f"Number of compounds: {len(scores)}")
    print(f"Number of actives: {sum(labels)}")
    print(f"Threshold {threshold}% selects top {int(np.ceil(len(scores) * threshold/100))} compounds")
    print(f"Score range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
    
    # Print first few compounds
    print("\nFirst few compounds (Score, Activity):")
    sorted_idx = np.argsort(scores)[::-1]
    for i in range(min(5, len(scores))):
        print(f"{scores[sorted_idx[i]]:.3f}, {labels[sorted_idx[i]]}")

def print_comparison(name: str, manual_value: float, analyzer_value: float, metric: str):
    """Print a formatted comparison of manual vs analyzer results."""
    print(f"\n{name} - {metric}:")
    print(f"Manual calculation: {manual_value:.6f}")
    print(f"Analyzer result:    {analyzer_value:.6f}")
    print(f"Difference:         {abs(manual_value - analyzer_value):.6f}")

@pytest.fixture
def test_data_files():
    """Get test data files from the test_data directory."""
    scores_file = dockm8_path / "test_data" / "performance" / "analyzer" / "test_scores.csv"
    activities_file = dockm8_path / "test_data" / "performance" / "analyzer" / "test_activities.csv"
    return scores_file, activities_file

@pytest.fixture
def analyzer_results(test_data_files):
    """Run analyzer once and cache results for all tests."""
    scores_file, activities_file = test_data_files
    results = run_consensus_analysis(
        scoring_data_path=scores_file,
        activity_data_path=activities_file,
        thresholds=[1, 2, 5],
        include_single=True
    )
    return results

def test_single_scoring_function(test_data_files, analyzer_results):
    """Test single scoring function analysis with careful data alignment."""
    scores_file, activities_file = test_data_files
    
    # Load test data
    scores_df = pd.read_csv(scores_file)
    activities_df = pd.read_csv(activities_file)
    
    # Ensure activities align with scores
    merged_data = pd.merge(
        scores_df[["ID", "KORP-PL"]],
        activities_df[["ID", "Activity"]],
        on="ID",
        how="inner"
    )
    print(f"\nTesting KORP-PL with {len(merged_data)} compounds")
    
    # Get scores and activities in aligned order
    scores = merged_data["KORP-PL"].values
    labels = merged_data["Activity"].values
    
    # Normalize scores
    scoring_info = RESCORING_FUNCTIONS.get("KORP-PL", {"best_value": "min"})
    normalized_scores = normalize_scores(
        scores,
        best_value=scoring_info["best_value"]
    )
    
    # Print data overview
    print_data_info(normalized_scores, labels, 1.0)
    
    # Calculate metrics manually - use percentages for thresholds
    manual_metrics = calculate_metrics(
        scores=normalized_scores,
        labels=labels,
        percentile=[1.0, 2.0, 5.0]  # Percentages
    )
    
    # Get analyzer results
    analyzer_korp = analyzer_results[analyzer_results["scoring_function"] == "KORP-PL"]
    
    # Compare metrics at each threshold
    for threshold, metrics in manual_metrics.items():
        print(f"\nThreshold: {threshold}%")
        analyzer_results_thresh = analyzer_korp[analyzer_korp["threshold"] == threshold]
        
        for metric, value in metrics.items():
            analyzer_value = analyzer_results_thresh[metric].iloc[0]
            print_comparison("KORP-PL", value, analyzer_value, metric)
            if metric in {"ccr", "mcc", "bedroc", "auc_roc", "aupr"}:
                assert np.isclose(analyzer_value, value, atol=0.05)
            else:
                assert np.isclose(analyzer_value, value, rtol=0.05)

def test_consensus_methods(test_data_files, analyzer_results):
    """Test consensus methods with proper consensus-activity alignment."""
    scores_file, activities_file = test_data_files
    
    # Load data
    scores_df = pd.read_csv(scores_file)
    activities_df = pd.read_csv(activities_file)
    
    print("\nTesting consensus methods for KORP-PL + ConvexPLR")
    
    # Prepare consensus input data
    consensus_data = pd.DataFrame()
    consensus_data["ID"] = scores_df["ID"]
    
    # Normalize input scores
    for col in ["KORP-PL", "ConvexPLR"]:
        scoring_info = RESCORING_FUNCTIONS.get(col, {"best_value": "min"})
        consensus_data[col] = normalize_scores(
            scores_df[col].values,
            best_value=scoring_info["best_value"]
        )
    
    for method in _METHODS:
        print(f"\nTesting {method.upper()} consensus method")
        
        # Calculate consensus scores
        consensus_result = apply_consensus_scoring(
            data=consensus_data,
            methods=[method],
            id_column="ID",
            normalize=False  # Already normalized
        )
        
        # After consensus calculation, merge with activities
        score_col = [col for col in consensus_result.columns if col != "ID"][0]
        merged_data = pd.merge(
            consensus_result,
            activities_df[["ID", "Activity"]],
            on="ID",
            how="inner"
        )
        
        consensus_scores = merged_data[score_col].values
        labels = merged_data["Activity"].values
        
        # Print data overview
        print_data_info(consensus_scores, labels, 1.0)
        
        # Calculate metrics
        manual_metrics = calculate_metrics(
            scores=consensus_scores,
            labels=labels,
            percentile=[1.0]  # Percentage
        )
        
        # Get analyzer results
        analyzer_results_method = analyzer_results[
            (analyzer_results["combination"] == "KORP-PL+ConvexPLR") &
            (analyzer_results["consensus_method"] == method) &
            (analyzer_results["threshold"] == 1.0)
        ]
        
        # Compare metrics
        for metric, value in manual_metrics[1.0].items():
            analyzer_value = analyzer_results_method[metric].iloc[0]
            print_comparison(f"{method.upper()}", value, analyzer_value, metric)
            if metric in {"ccr", "mcc", "bedroc", "auc_roc", "aupr"}:
                assert np.isclose(analyzer_value, value, atol=0.05)
            else:
                assert np.isclose(analyzer_value, value, rtol=0.05)

def test_three_way_combination(test_data_files, analyzer_results):
    """Test three-way combinations with proper consensus-activity alignment."""
    scores_file, activities_file = test_data_files
    
    # Load data
    scores_df = pd.read_csv(scores_file)
    activities_df = pd.read_csv(activities_file)
    
    print("\nTesting three-way combination: KORP-PL + ConvexPLR + AD4")
    
    # Prepare consensus input data
    consensus_data = pd.DataFrame()
    consensus_data["ID"] = scores_df["ID"]
    
    # Normalize input scores
    for col in ["KORP-PL", "ConvexPLR", "AD4"]:
        scoring_info = RESCORING_FUNCTIONS.get(col, {"best_value": "min"})
        consensus_data[col] = normalize_scores(
            scores_df[col].values,
            best_value=scoring_info["best_value"]
        )
    
    # Calculate consensus
    consensus_result = apply_consensus_scoring(
        data=consensus_data,
        methods=["ecr"],
        id_column="ID",
        normalize=False  # Already normalized
    )
    
    # After consensus calculation, merge with activities
    score_col = [col for col in consensus_result.columns if col != "ID"][0]
    merged_data = pd.merge(
        consensus_result,
        activities_df[["ID", "Activity"]],
        on="ID",
        how="inner"
    )
    
    consensus_scores = merged_data[score_col].values
    labels = merged_data["Activity"].values
    
    # Print data overview
    print_data_info(consensus_scores, labels, 1.0)
    
    # Calculate metrics
    manual_metrics = calculate_metrics(
        scores=consensus_scores,
        labels=labels,
        percentile=[1.0]  # Percentage
    )
    
    # Get analyzer results
    analyzer_results_three = analyzer_results[
        (analyzer_results["combination"] == "KORP-PL+ConvexPLR+AD4") &
        (analyzer_results["consensus_method"] == "ecr") &
        (analyzer_results["threshold"] == 1.0)
    ]
    
    # Compare metrics
    for metric, value in manual_metrics[1.0].items():
        analyzer_value = analyzer_results_three[metric].iloc[0]
        print_comparison("Three-way ECR", value, analyzer_value, metric)
        if metric in {"ccr", "mcc", "bedroc", "auc_roc", "aupr"}:
            assert np.isclose(analyzer_value, value, atol=0.05)
        else:
            assert np.isclose(analyzer_value, value, rtol=0.05)

def test_all_single_functions(test_data_files, analyzer_results):
    """Test all individual scoring functions with careful data alignment."""
    scores_file, activities_file = test_data_files
    
    # Load and merge data to ensure alignment
    scores_df = pd.read_csv(scores_file)
    activities_df = pd.read_csv(activities_file)
    
    scoring_functions = ["KORP-PL", "ConvexPLR", "AD4", "RFScoreVS"]
    
    for func in scoring_functions:
        print(f"\nTesting {func}")
        
        # Merge scores with activities
        merged_data = pd.merge(
            scores_df[["ID", func]],
            activities_df[["ID", "Activity"]],
            on="ID",
            how="inner"
        )
        
        scores = merged_data[func].values
        labels = merged_data["Activity"].values
        
        # Normalize scores
        scoring_info = RESCORING_FUNCTIONS.get(func, {"best_value": "min"})
        normalized_scores = normalize_scores(
            scores,
            best_value=scoring_info["best_value"]
        )
        
        # Print data overview
        print_data_info(normalized_scores, labels, 1.0)
        
        # Calculate metrics manually
        manual_metrics = calculate_metrics(
            scores=normalized_scores,
            labels=labels,
            percentile=[1.0]  # Percentage
        )
        
        # Get analyzer results
        analyzer_results_func = analyzer_results[
            (analyzer_results["scoring_function"] == func) &
            (analyzer_results["threshold"] == 1.0)
        ]
        
        # Compare metrics
        for metric, value in manual_metrics[1.0].items():
            analyzer_value = analyzer_results_func[metric].iloc[0]
            print_comparison(func, value, analyzer_value, metric)
            assert np.isclose(analyzer_value, value, rtol=1e-3)
