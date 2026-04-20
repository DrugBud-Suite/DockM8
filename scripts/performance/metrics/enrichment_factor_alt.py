# metricm8/metrics/enrichment_factor.py
import numpy as np
import sys
from pathlib import Path

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.performance.utils import validate_inputs, prepare_screening_data


def calculate_enrichment_factor_alt(
    scores: np.ndarray, labels: np.ndarray, percentile: float | list[float]
) -> float | list[float]:
    """
    Calculate Enrichment Factor (EF) for virtual screening results.

    EF = (n_actives_selected / n_selected) / (n_actives_total / n_total)

    Parameters
    ----------
    scores : np.ndarray
        Screening scores
    labels : np.ndarray
        Binary labels (0/1)
    percentile : Union[float, List[float]]
        Percentile threshold(s) for selection (e.g., 1.0 for top 1%)

    Returns:
    -------
    Union[float, List[float]]
        Enrichment factor value(s)

    Notes:
    -----
    The enrichment factor represents how many times better the selection
    method is compared to random selection. An EF of 1 means the method
    is no better than random, while a value of 10 means it is 10 times
    better than random selection.
    """
    percentiles = np.atleast_1d(percentile)
    validate_inputs(scores, labels, percentiles)

    results = []

    for p in percentiles:
        # Get sorted data and selection size
        sorted_labels, sorted_scores, n_selected = prepare_screening_data(scores, labels, p)

        # Calculate number of actives in selection
        N = len(sorted_labels)  # Total number of compounds
        Ns = len(sorted_scores[:n_selected])  # Number of compounds in selection
        n = np.sum(sorted_labels)  # Total number of actives
        ns = np.sum(sorted_labels[:n_selected])  # Number of actives in selection

        # Calculate enrichment factor
        ef = (ns / Ns) * (N / n)

        results.append(round(ef, 3))

    return results[0] if len(percentiles) == 1 else results
