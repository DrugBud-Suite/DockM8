import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.consensus.consensus import (
    apply_consensus_scoring,
    load_and_validate_data,
    handle_nan_values,
)


@pytest.fixture
def test_data():
    """Load test data from CSV file."""
    df = pd.read_csv(str(dockm8_path / "test_data" / "consensus" / "allposes_rescored.csv"))
    df["ID"] = df["Pose ID"].str.split("_").str[0]
    return df


def test_available_methods():
    """Test that expected consensus methods are available."""
    methods = get_available_methods()
    expected_methods = ["ecr", "rbr", "rbv", "zscore", "pareto"]
    assert sorted(methods) == sorted(expected_methods)


def test_load_and_validate_data(test_data):
    """Test data loading and validation."""
    data, valid_cols = load_and_validate_data(test_data, "ID")

    # All columns except Pose ID and ID should be valid
    expected_cols = [
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
        "RTMScore",
    ]
    assert set(valid_cols) == set(expected_cols)
    assert "Pose ID" not in valid_cols
    assert "ID" not in valid_cols


def test_consensus_methods_individual(test_data):
    """Test each consensus method individually."""
    methods = get_available_methods()

    for method in methods:
        result = apply_consensus_scoring(data=test_data, methods=method, id_column="ID", normalize=True)

        # Verify basic properties
        assert isinstance(result, pd.DataFrame)
        assert "ID" in result.columns
        # Check if method exists in columns (case-insensitive)
        assert any(
            col.lower() == method.lower() for col in result.columns
        ), f"Method {method} not found in result columns (case-insensitive)"
        assert len(result) == len(test_data["ID"].unique())

        # Get the actual column name (case-insensitive match)
        method_col = next(col for col in result.columns if col.lower() == method.lower())
        assert all(result[method_col].between(0, 1))


def test_consensus_aggregation(test_data):
    """Test different aggregation methods."""
    # Test best aggregation
    best_result = apply_consensus_scoring(
        data=test_data, methods="zscore", id_column="ID", normalize=False, aggregation="best"
    )

    # Test average aggregation
    avg_result = apply_consensus_scoring(
        data=test_data, methods="zscore", id_column="ID", normalize=False, aggregation="avg"
    )

    # Basic checks
    assert len(best_result) == len(test_data["ID"].unique())
    assert len(avg_result) == len(test_data["ID"].unique())

    # Get actual column name (case-insensitive)
    score_col = next(col for col in best_result.columns if col.lower() == "zscore")

    # Find molecules with multiple poses
    multi_pose_ids = test_data["ID"].value_counts()[test_data["ID"].value_counts() > 1].index
    assert len(multi_pose_ids) > 0, "No molecules with multiple poses found in test data"

    # Create a detailed comparison
    print("\nDetailed comparison of best vs average scores:")
    print("-" * 60)
    differences_found = False

    for mol_id in multi_pose_ids:
        best_score = best_result.loc[best_result["ID"] == mol_id, score_col].iloc[0]
        avg_score = avg_result.loc[avg_result["ID"] == mol_id, score_col].iloc[0]

        # Get all poses and their scores for this molecule
        mol_poses = test_data[test_data["ID"] == mol_id]["Pose ID"].tolist()

        print(f"\nMolecule {mol_id}:")
        print(f"Number of poses: {len(mol_poses)}")
        print(f"Poses: {', '.join(mol_poses)}")
        print(f"Best score: {best_score:.6f}")
        print(f"Avg score: {avg_score:.6f}")
        print(f"Difference: {abs(best_score - avg_score):.6f}")

        if not np.isclose(best_score, avg_score, rtol=1e-5):
            differences_found = True

    print("\n" + "-" * 60)

    assert differences_found, (
        "Best and average aggregation produced identical scores for all molecules. "
        "These methods should produce different results for molecules with multiple poses."
    )


def test_multiple_methods(test_data):
    """Test using multiple consensus methods together."""
    methods = ["zscore", "rbr"]
    result = apply_consensus_scoring(data=test_data, methods=methods, id_column="ID")

    # Check that all methods are in the result (case-insensitive)
    for method in methods:
        assert any(
            col.lower() == method.lower() for col in result.columns
        ), f"Method {method} not found in result columns (case-insensitive)"

    assert len(result) == len(test_data["ID"].unique())

    # Verify scores are normalized
    for method in methods:
        method_col = next(col for col in result.columns if col.lower() == method.lower())
        assert all(result[method_col].between(0, 1))


def test_output_file(test_data, tmp_path):
    """Test saving results to file."""
    output_file = tmp_path / "results.csv"

    result_path = apply_consensus_scoring(data=test_data, methods="zscore", id_column="ID", output=output_file)

    assert result_path.exists()
    loaded_results = pd.read_csv(result_path)
    assert any(col.lower() == "zscore" for col in loaded_results.columns)
    assert len(loaded_results) == len(test_data["ID"].unique())


def test_invalid_inputs(test_data):
    """Test error handling for invalid inputs."""
    # Test invalid data type
    bad_df = pd.DataFrame(
        {
            "Pose ID": ["mol1_pose1", "mol2_pose1"],
            "ID": ["mol1", "mol2"],
            "score": ["not a number", "also not a number"],
        }
    )

    with pytest.raises(ValueError):
        apply_consensus_scoring(bad_df, methods="zscore")

    # Test invalid method
    with pytest.raises(ValueError):
        apply_consensus_scoring(test_data, methods="invalid_method")


def test_nan_handling(test_data):
    """Test NaN handling strategies."""
    data_with_nans = test_data.copy()
    data_with_nans.loc[0, "KORP-PL"] = None
    data_with_nans.loc[1, "CHEMPLP"] = None

    strategies = ["drop", "fill_mean", "fill_median"]
    for strategy in strategies:
        result = apply_consensus_scoring(data=data_with_nans, methods="zscore", id_column="ID", nan_strategy=strategy)
        score_col = next(col for col in result.columns if col.lower() == "zscore")
        assert not result[score_col].isna().any()
