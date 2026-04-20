import numpy as np
from rdkit.ML.Scoring.Scoring import CalcAUC
import sys
from pathlib import Path

# Search for 'DockM8' in parent directories
tests_path = next((p / "tests" for p in Path(__file__).resolve().parents if (p / "tests").is_dir()), None)
dockm8_path = tests_path.parent
sys.path.append(str(dockm8_path))

from scripts.performance.utils import prepare_scores_for_rdkit


def calculate_rdkitauc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate AUC using RDKit's implementation.

    Parameters
    ----------
    scores : np.ndarray
        Screening scores
    labels : np.ndarray
        Binary labels (0/1)

    Returns:
    -------
    float
        AUC score between 0 and 1
    """
    rdkit_data = prepare_scores_for_rdkit(scores, labels)
    return round(CalcAUC(rdkit_data, 1), 3)
