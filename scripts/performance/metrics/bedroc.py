import numpy as np
from rdkit.ML.Scoring.Scoring import CalcBEDROC

import sys
from pathlib import Path

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.performance.utils import prepare_scores_for_rdkit


def calculate_bedroc(scores: np.ndarray, labels: np.ndarray, alpha: float = 80.5) -> float:
    """
    Calculate Boltzmann-Enhanced Discrimination of ROC (BEDROC).

    Parameters
    ----------
    scores : np.ndarray
        Screening scores
    labels : np.ndarray
        Binary labels (0/1)
    alpha : float, optional
        Parameter for early recognition, default=80.5

    Returns:
    -------
    float
        BEDROC score between 0 and 1
    """
    # Normalize scores and prepare data in RDKit format
    # scores = normalize_scores(scores)
    rdkit_data = prepare_scores_for_rdkit(scores, labels)
    return round(CalcBEDROC(rdkit_data, 1, alpha), 3)
