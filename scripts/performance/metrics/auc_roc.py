import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate Area Under the ROC Curve.

    Parameters
    ----------
    scores : np.ndarray
        Screening scores
    labels : np.ndarray
        Binary labels (0/1)

    Returns:
    -------
    float
        AUC-ROC score between 0 and 1
    """
    return round(roc_auc_score(labels, scores, multi_class="ovo"), 3)
