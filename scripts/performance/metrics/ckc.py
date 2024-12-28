# metricm8/metrics/ckc.py
import numpy as np

import sys
from pathlib import Path

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.performance.utils import validate_inputs, prepare_screening_data


def calculate_ckc(scores: np.ndarray, labels: np.ndarray, percentile: float | list[float]) -> float | list[float]:
    """
    Calculate Cohen's Kappa Coefficient (CKC) for virtual screening results.

    CKC = (po - pe) / (1 - pe)
    where po = observed agreement
                  pe = expected agreement by chance

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
    CKC value(s) between -1 and 1
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

        numerator = N * n + N * Ns - 2 * ns * N
        denominator = N * n + N * Ns - 2 * n * Ns

        # Calculate Cohen's Kappa
        if denominator == 0:
            ckc = 0
        else:
            ckc = 1 - (numerator / denominator)

        results.append(round(ckc, 3))

    return results[0] if len(percentiles) == 1 else results
