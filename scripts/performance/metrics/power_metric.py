import numpy as np
from scripts.performance.utils import prepare_screening_data, validate_inputs


def calculate_power_metric(
    scores: np.ndarray, labels: np.ndarray, percentile: float | list[float]
) -> float | list[float]:
    """
    Calculate Power Metric (PM) for virtual screening results.

    PM = (ns * N - n * ns)/(ns * N - 2 * n * ns + n * Ns)

    where:
    ns = number of actives in selection
    n = total number of actives
    N = total number of compounds
    Ns = number of compounds in selection

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
        Power metric value(s)
    """
    percentiles = np.atleast_1d(percentile)
    validate_inputs(scores, labels, percentiles)

    results = []
    n_total = len(scores)  # N
    n_actives_total = np.sum(labels)  # n

    if n_actives_total == 0:
        raise ValueError("No active compounds in the dataset")

    for p in percentiles:
        sorted_labels, sorted_scores, n_selected = prepare_screening_data(scores, labels, p)

        # Calculate values needed for formula
        N = len(sorted_labels)  # total compounds
        Ns = n_selected  # compounds in selection
        n = np.sum(sorted_labels)  # total actives
        ns = np.sum(sorted_labels[:n_selected])  # actives in selection

        # Calculate power metric using exact formula from paper
        numerator = ns * N - n * ns
        denominator = ns * N - 2 * n * ns + n * Ns

        # Handle division by zero
        if denominator == 0:
            pm = 0
        else:
            pm = numerator / denominator

        results.append(round(pm, 3))

    return results[0] if len(percentiles) == 1 else results
