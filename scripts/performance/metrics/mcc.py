# metricm8/metrics/mcc.py
import numpy as np
import sys
from pathlib import Path

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.performance.utils import validate_inputs, prepare_screening_data


def calculate_mcc(scores: np.ndarray, labels: np.ndarray, percentile: float | list[float]) -> float | list[float]:
    """
    Calculate Matthews Correlation Coefficient (MCC) for virtual screening results.

    MCC = (TP × TN - FP × FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))

    Parameters
    ----------
    scores : np.ndarray
        Screening scores
    labels : np.ndarray
        Binary labels (0/1)
    percentile : Union[float, List[float]]
        Percentile threshold(s) for selection

    Returns:
    -------
    Union[float, List[float]]
        MCC value(s)
    """
    percentiles = np.atleast_1d(percentile)
    validate_inputs(scores, labels, percentiles)

    results = []
    n_total = len(scores)
    n_actives_total = np.sum(labels)

    for p in percentiles:
        sorted_labels, sorted_scores, n_selected = prepare_screening_data(scores, labels, p)

        # Calculate number of actives in selection
        N = len(sorted_labels)  # Total number of compounds
        Ns = len(sorted_scores[:n_selected])  # Number of compounds in selection
        n = np.sum(sorted_labels)  # Total number of actives
        ns = np.sum(sorted_labels[:n_selected])  # Number of actives in selection

        # Calculate MCC
        numerator = N * ns - Ns * n
        denominator = np.sqrt(Ns * n * (N - n) * (N - Ns))

        # Handle edge cases
        if denominator == 0:
            mcc = 0
        else:
            mcc = numerator / denominator

        results.append(round(mcc, 3))

    return results[0] if len(percentiles) == 1 else results
