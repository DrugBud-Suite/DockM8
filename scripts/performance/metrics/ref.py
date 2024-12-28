# metricm8/metrics/ref.py
import sys
from pathlib import Path

import numpy as np

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.performance.utils import prepare_screening_data, validate_inputs


def calculate_relative_enrichment_factor(
    scores: np.ndarray, labels: np.ndarray, percentile: float | list[float]
) -> float | list[float]:
    """
    Calculate Relative Enrichment Factor (REF) for virtual screening results.

    REF = 100 * (n_actives_selected / min(N * χ, n_actives_total))

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
        Relative enrichment factor value(s)

    Notes:
    -----
    REF addresses the saturation effect in EF and has well-defined
    boundaries (0-100). A REF of 100 means perfect selection of actives,
    while 0 means no actives were selected.
    """
    percentiles = np.atleast_1d(percentile)
    validate_inputs(scores, labels, percentiles)

    results = []
    n_total = len(scores)
    n_actives_total = np.sum(labels)

    # Avoid division by zero
    if n_actives_total == 0:
        raise ValueError("No active compounds in the dataset")

    for p in percentiles:
        # Get sorted data and selection size
        sorted_labels, sorted_scores, n_selected = prepare_screening_data(scores, labels, p)

        # Calculate number of actives in selection
        N = len(sorted_labels)  # Total number of compounds
        Ns = len(sorted_scores[:n_selected])  # Number of compounds in selection
        n = np.sum(sorted_labels)  # Total number of actives
        ns = np.sum(sorted_labels[:n_selected])  # Number of actives in selection

        # Calculate maximum possible actives at this threshold
        max_possible_actives = min(n_selected, n)

        # Calculate relative enrichment factor (as percentage)
        ref = (100 * ns) / max_possible_actives
        results.append(round(ref, 3))

    return results[0] if len(percentiles) == 1 else results
