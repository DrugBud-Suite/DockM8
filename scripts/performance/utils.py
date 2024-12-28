# metricm8/core/calculate.py
import sys
from pathlib import Path

import numpy as np

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))


def validate_inputs(scores: np.ndarray, labels: np.ndarray, percentile: float | list[float]) -> None:
    """Input validation as before..."""
    if np.any((np.atleast_1d(percentile) <= 0) | (np.atleast_1d(percentile) > 100)):
        msg = "Percentile thresholds must be between 0 and 100"
        raise ValueError(msg)


def prepare_screening_data(
    scores: np.ndarray, labels: np.ndarray, percentile: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Prepare screening data by sorting and applying percentile threshold.

    Parameters
    ----------
    scores : np.ndarray
        Screening scores
    labels : np.ndarray
        Binary labels (0/1)
    percentile : float
        Percentile threshold (1 = top 1% etc.)

    Returns:
    -------
    sorted_labels : np.ndarray
        Labels sorted by descending scores
    sorted_scores : np.ndarray
        Sorted scores
    n_selected : int
        Number of compounds selected at percentile threshold
    """
    # Create index array to maintain score-label correspondence
    indices = np.arange(len(scores))

    # Sort by scores in descending order, keeping track of indices
    sorted_indices = scores.argsort()[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Calculate number of compounds to select
    n_selected = int(np.ceil(len(scores) * percentile / 100))

    return sorted_labels, sorted_scores, n_selected


def prepare_scores_for_rdkit(scores: np.ndarray, labels: np.ndarray, sort: bool = True) -> list[list[float | int]]:
    """
    Convert separate score and label arrays into RDKit's expected format. RDKit expects a list of [score, label] pairs.

    Parameters
    ----------
    scores : np.ndarray
        Array of screening scores
    labels : np.ndarray
        Array of binary labels (0/1)
    sort : bool, optional
        Whether to sort scores in descending order (default: True)

    Returns:
    -------
    List[List[Union[float, int]]]
        List of [score, label] pairs, optionally sorted by descending scores
    """
    if sort:
        # Sort by scores in descending order
        sorted_indices = scores.argsort()[::-1]
        sorted_scores = scores[sorted_indices]
        sorted_labels = labels[sorted_indices]
        return list(zip(sorted_scores, sorted_labels, strict=False))

    return list(zip(scores, labels, strict=False))


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to range [0, 1].

    Parameters
    ----------
    scores : np.ndarray
        Array of screening scores

    Returns:
    -------
    np.ndarray
        Normalized scores
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)
