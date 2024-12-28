# metricm8/metrics/roce.py
import sys
from pathlib import Path

import numpy as np

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.performance.utils import prepare_screening_data, validate_inputs


def calculate_roce(scores: np.ndarray, labels: np.ndarray, percentile: float | list[float]) -> float | list[float]:
    """
    Calculate ROC Enrichment (ROCE) for virtual screening results.

    ROCE = (TPR/FPR) = (n_actives_selected/n_actives_total) /
                                           ((n_selected - n_actives_selected)/(n_total - n_actives_total))

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
    ROCE value(s)
    """
    percentiles = np.atleast_1d(percentile)
    validate_inputs(scores, labels, percentiles)

    results = []
    n_total = len(scores)
    n_actives_total = np.sum(labels)

    if n_actives_total == 0:
        raise ValueError("No active compounds in the dataset")

    for p in percentiles:
        sorted_labels, sorted_scores, n_selected = prepare_screening_data(scores, labels, p)

        # Calculate number of actives in selection
        N = len(sorted_labels)  # Total number of compounds
        Ns = len(sorted_scores[:n_selected])  # Number of compounds in selection
        n = np.sum(sorted_labels)  # Total number of actives
        ns = np.sum(sorted_labels[:n_selected])  # Number of actives in selection

        # Avoid division by zero
        if (n * (Ns - ns)) == 0:
            roce = np.inf
        else:
            roce = (ns * (N - n)) / (n * (Ns - ns))

        results.append(round(roce, 3))

    return results[0] if len(percentiles) == 1 else results
