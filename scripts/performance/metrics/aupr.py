from sklearn.metrics import average_precision_score
import numpy as np


def calculate_aupr(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Area Under the Precision-Recall Curve.

    Parameters
    ----------
    scores : np.ndarray
        Screening scores
    labels : np.ndarray
        Binary labels (0/1)

    Returns:
    -------
    float
        AUPR score between 0 and 1
    """
    return round(average_precision_score(labels, scores), 3)
