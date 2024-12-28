import sys
from pathlib import Path
import numpy as np
import pytest
import pandas as pd
from tabulate import tabulate

from .test_data import generate_perfect_ranking, generate_worst_ranking, generate_exponential_ranking

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.performance.calculate_metrics import calculate_metrics

def format_value_with_std(mean, std):
    """Format mean ± std with appropriate precision"""
    if std == 0:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def print_metric_table(stats, metric):
    """Print paper-style table for a specific metric"""
    # Create table headers
    headers = ["λ", "0.5%", "1.0%", "2.0%"]

    # Create table rows
    rows = []
    # Add Perfect and Worst rows
    for ranking in ["Perfect", "Worst"]:
        row = [ranking]
        for percentile in [0.5, 1.0, 2.0]:
            row.append(
                format_value_with_std(
                    stats[metric][ranking][percentile]["mean"], stats[metric][ranking][percentile]["std"]
                )
            )
        rows.append(row)

    # Add Lambda rows
    for lambda_val in [2, 5, 10, 20, 40]:
        row = [str(lambda_val)]
        for percentile in [0.5, 1.0, 2.0]:
            row.append(
                format_value_with_std(
                    stats[metric][f"Lambda{lambda_val}"][percentile]["mean"],
                    stats[metric][f"Lambda{lambda_val}"][percentile]["std"],
                )
            )
        rows.append(row)

    print(f"\n{metric.upper()}:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def parse_mean_std(value):
    """Parse a string of format 'mean±std' into (mean, std) tuple"""
    try:
        parts = value.split("\xb1")  # Split on ± character
        if len(parts) == 2:
            return round(float(parts[0]), 2), round(float(parts[1]), 2)
        return round(float(value), 2), 0.0
    except:
        return 0.0, 0.0


def calculate_perfect_ranking_expectations(n: int, N: int, percentiles: list[float]) -> dict:
    """Calculate expected metric values for perfect ranking scenario."""
    expected = {"pm": {}, "ef": {}, "ref": {}, "roce": {}, "ccr": {}, "mcc": {}, "ckc": {}}

    for p in percentiles:
        # Calculate number of selected compounds
        Ns = int(np.ceil(N * p / 100))

        # Number of actives in selection for perfect ranking
        ns = min(Ns, n)

        # Power Metric
        numerator = ns * N - n * ns
        denominator = ns * N - 2 * n * ns + n * Ns
        expected["pm"][p] = numerator / denominator if denominator != 0 else 0

        # Enrichment Factor
        expected["ef"][p] = (N * ns) / (n * Ns)

        # Relative Enrichment Factor
        max_possible = min(Ns, n)
        expected["ref"][p] = 100 * (ns / max_possible)

        # ROCE
        if Ns <= n:
            expected["roce"][p] = float("inf")
        else:
            expected["roce"][p] = (ns * (N - n)) / (n * (Ns - ns))

        # CCR
        tpr = ns / n  # True Positive Rate
        tnr = (N - max(Ns, n)) / (N - n)  # True Negative Rate
        expected["ccr"][p] = 0.5 * (tpr + tnr)

        # MCC
        # For perfect ranking with exact selection
        if Ns == n:
            expected["mcc"][p] = 1.0
        else:
            numerator = N * ns - Ns * n
            denominator = np.sqrt(Ns * n * (N - n) * (N - Ns))
            expected["mcc"][p] = numerator / denominator if denominator != 0 else 0

        # Cohen's Kappa Coefficient (CKC)
        numerator = N * n + N * Ns - 2 * ns * N
        denominator = N * n + N * Ns - 2 * n * Ns
        if denominator == 0:
            expected["ckc"][p] = 0
        else:
            expected["ckc"][p] = 1 - (numerator / denominator)

    return expected


class TestMetricsSyntheticData:
    """Test metrics using synthetic rankings and calculate_metrics function"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test data with different ranking scenarios"""
        self.n_actives = 100
        self.n_total = 10000
        self.percentiles = [0.5, 1.0, 2.0]
        self.lambda_values = [2, 5, 10, 20, 40]
        self.metrics = ["pm", "ef", "ref", "roce", "ccr", "mcc", "ckc"]
        self.n_repeats = 10

        # Load paper data
        try:
            self.paper_data = pd.read_csv(
                dockm8_path / "test_data/performance/metrics/paper_data.csv",
                encoding="latin1",  # Use latin1 encoding to handle ± character
            )
        except Exception as e:
            print(f"Error reading paper data: {e}")
            self.paper_data = pd.DataFrame()

        # Parse mean and std from paper data
        self.paper_values = {}
        for _, row in self.paper_data.iterrows():
            metric = row["Metric"].lower()
            threshold = row["Threshold"]

            if metric not in self.paper_values:
                self.paper_values[metric] = {}
            if threshold not in self.paper_values[metric]:
                self.paper_values[metric][threshold] = {}

            for lambda_val in self.lambda_values:
                mean, std = parse_mean_std(row[f"Lambda{lambda_val}"])
                self.paper_values[metric][threshold][lambda_val] = {"mean": mean, "std": std}

        # Store results for all repeats
        self.results_all_repeats = []

        for repeat in range(self.n_repeats):
            np.random.seed(42 + repeat)
            results = {}

            # Perfect and worst rankings (deterministic)
            if repeat == 0:
                perfect_scores, perfect_labels = generate_perfect_ranking(self.n_actives, self.n_total)
                worst_scores, worst_labels = generate_worst_ranking(self.n_actives, self.n_total)

                results["Perfect"] = calculate_metrics(perfect_scores, perfect_labels, self.percentiles, self.metrics)
                results["Worst"] = calculate_metrics(worst_scores, worst_labels, self.percentiles, self.metrics)

            # Lambda-based rankings
            for lambda_val in self.lambda_values:
                scores, labels = generate_exponential_ranking(self.n_actives, self.n_total, lambda_val)
                results[f"Lambda{lambda_val}"] = calculate_metrics(scores, labels, self.percentiles, self.metrics)

            self.results_all_repeats.append(results)

        # Calculate statistics and store in a structured way
        self.stats = {}
        for metric in self.metrics:
            self.stats[metric] = {"Perfect": {}, "Worst": {}}
            for lambda_val in self.lambda_values:
                self.stats[metric][f"Lambda{lambda_val}"] = {}

            # Process each percentile
            for percentile in self.percentiles:
                # Perfect and Worst (deterministic)
                self.stats[metric]["Perfect"][percentile] = {
                    "mean": round(self.results_all_repeats[0]["Perfect"][percentile][metric], 2),
                    "std": 0.0,
                }
                self.stats[metric]["Worst"][percentile] = {
                    "mean": round(self.results_all_repeats[0]["Worst"][percentile][metric], 2),
                    "std": 0.0,
                }

                # Lambda-based results
                for lambda_val in self.lambda_values:
                    values = [
                        repeat_results[f"Lambda{lambda_val}"][percentile][metric]
                        for repeat_results in self.results_all_repeats
                    ]
                    self.stats[metric][f"Lambda{lambda_val}"][percentile] = {
                        "mean": round(np.mean(values), 2),
                        "std": round(np.std(values), 2),
                    }

    def test_metrics_against_paper(self):
        """Test metrics against paper values"""
        for metric in self.metrics:
            print_metric_table(self.stats, metric)
            print(f"\nTesting {metric.upper()} against paper values:")

            if metric not in self.paper_values:
                print(f"No paper data available for {metric}")
                continue

            for threshold in self.percentiles:
                if threshold not in self.paper_values[metric]:
                    continue

                print(f"\nThreshold {threshold}%:")
                for lambda_val in self.lambda_values:
                    # Get paper values
                    paper_stats = self.paper_values[metric][threshold][lambda_val]
                    paper_mean = paper_stats["mean"]
                    paper_std = paper_stats["std"]

                    # Get our values
                    stats = self.stats[metric][f"Lambda{lambda_val}"][threshold]
                    mean_val = stats["mean"]
                    std_val = stats["std"]

                    # Define acceptable range from paper
                    paper_min = round(paper_mean - paper_std, 2)
                    paper_max = round(paper_mean + paper_std, 2)

                    # Round mean_val for comparison
                    mean_val = round(mean_val, 2)

                    # Check if our mean falls within paper's range
                    assert (
                        paper_min <= mean_val <= paper_max
                    ), f"{metric} for lambda={lambda_val}: {mean_val:.2f} falls outside paper range [{paper_min:.2f}, {paper_max:.2f}]"

                    # Print comparison
                    print(
                        f"λ={lambda_val:2d}: {mean_val:.2f}±{std_val:.2f} (paper: {paper_mean:.2f}±{paper_std:.2f}, range: [{paper_min:.2f}, {paper_max:.2f}])"
                    )

    def test_perfect_ranking_properties(self):
        """Test that perfect ranking achieves expected results"""
        print("\nTesting perfect ranking properties:")

        expected_perfect = calculate_perfect_ranking_expectations(self.n_actives, self.n_total, self.percentiles)

        for metric in self.metrics:
            print(f"\nTesting {metric.upper()} perfect ranking properties:")

            for percentile in self.percentiles:
                stats = self.stats[metric]["Perfect"][percentile]
                mean_val = round(stats["mean"], 2)
                std_val = round(stats["std"], 2)

                print(f"Percentile {percentile}%: {mean_val:.2f}±{std_val:.2f}")

                if metric == "roce":
                    # Special case for ROCE which should be infinity
                    expected = expected_perfect[metric][percentile]
                    if np.isinf(expected):
                        assert np.isinf(
                            mean_val
                        ), f"Perfect ranking {metric.upper()} should be infinity, got {mean_val:.2f}"
                    else:
                        assert (
                            mean_val == pytest.approx(expected, rel=0.1)
                        ), f"Perfect ranking {metric.upper()} at {percentile}% should be close to {expected:.2f}, got {mean_val:.2f}"
                elif metric == "ef":
                    expected = expected_perfect[metric][percentile]
                    assert (
                        mean_val == pytest.approx(expected, rel=0.1)
                    ), f"Perfect ranking {metric.upper()} at {percentile}% should be close to {expected:.2f}, got {mean_val:.2f}"
                elif metric == "ckc":
                    expected = expected_perfect[metric][percentile]
                    assert (
                        mean_val == pytest.approx(expected, abs=0.1)
                    ), f"Perfect ranking {metric.upper()} at {percentile}% should be close to {expected:.2f}, got {mean_val:.2f}"
                elif metric in ["pm", "ref", "ccr", "mcc"]:
                    expected = expected_perfect[metric][percentile]
                    assert mean_val == pytest.approx(
                        expected, abs=0.1
                    ), f"Perfect ranking {metric.upper()} should be > {expected:.2f}, got {mean_val:.2f}"

    def test_worst_ranking_properties(self):
        """Test that worst ranking achieves expected results"""
        print("\nTesting worst ranking properties:")

        expected_worst = {
            "pm": 0.0,  # Power Metric should be near 0
            "ef": 0.0,  # Enrichment Factor should be near 0
            "ref": 0.0,  # Relative Enrichment Factor should be near 0
            "roce": 0.0,  # ROC Enrichment should be near 0
            "ccr": 0.5,  # Correct Classification Rate should be near random (0.5)
            "mcc": 0.0,  # Matthews Correlation Coefficient should be near 0
            "ckc": 0.0,  # Cohen's Kappa Coefficient should be near 0
        }

        for metric in self.metrics:
            print(f"\nTesting {metric.upper()} worst ranking properties:")

            for percentile in self.percentiles:
                stats = self.stats[metric]["Worst"][percentile]
                mean_val = round(stats["mean"], 2)
                std_val = round(stats["std"], 2)

                print(f"Percentile {percentile}%: {mean_val:.2f}±{std_val:.2f}")

                if metric in ["pm", "ef", "ref", "roce", "mcc", "ckc"]:
                    assert (
                        mean_val == pytest.approx(expected_worst[metric], abs=0.1)
                    ), f"Worst ranking {metric.upper()} should be close to {expected_worst[metric]:.2f}, got {mean_val:.2f}"
                elif metric == "ccr":
                    assert (
                        mean_val == pytest.approx(expected_worst[metric], abs=0.1)
                    ), f"Worst ranking {metric.upper()} should be close to {expected_worst[metric]:.2f}, got {mean_val:.2f}"
