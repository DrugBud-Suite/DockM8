import sys
from pathlib import Path

import numpy as np
from rdkit.ML.Scoring.Scoring import CalcEnrichment

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.performance.utils import prepare_scores_for_rdkit


def calculate_rdkit_ef(scores: np.ndarray, labels: np.ndarray, percentile: float | list[float]) -> float:
    """
    Calculate RDKit Enrichment Factor.

    Parameters
    ----------
    scores : np.ndarray
        Screening scores
    labels : np.ndarray
        Binary labels (0/1)
    percentile : float | list[float]
        Percentile threshold(s) for selection

    Returns:
    -------
    float | list[float]
        Enrichment factor value(s)
    """
    results = []
    rdkit_data = prepare_scores_for_rdkit(scores, labels)
    percentiles = np.atleast_1d(percentile)
    fractions = [p / 100.0 for p in percentiles]

    for f in fractions:
        # CalcEnrichment returns a list of values, we need to extract the first one
        rdkit_ef = CalcEnrichment(rdkit_data, 1, [f])[0]
        results.append(round(rdkit_ef, 3))

    return results[0] if len(percentiles) == 1 else results
