# metricm8/core/calculate.py
import sys
from pathlib import Path

import numpy as np

# Search for 'DockM8' in parent directories
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.performance.metrics.auc_roc import calculate_auc_roc
from scripts.performance.metrics.aupr import calculate_aupr
from scripts.performance.metrics.bedroc import calculate_bedroc
from scripts.performance.metrics.ccr import calculate_ccr
from scripts.performance.metrics.enrichment_factor import calculate_enrichment_factor
from scripts.performance.metrics.enrichment_factor_alt import calculate_enrichment_factor_alt
from scripts.performance.metrics.mcc import calculate_mcc
from scripts.performance.metrics.power_metric import calculate_power_metric
from scripts.performance.metrics.rdkit_auc import calculate_rdkitauc
from scripts.performance.metrics.rdkit_ef import calculate_rdkit_ef
from scripts.performance.metrics.ref import calculate_relative_enrichment_factor
from scripts.performance.metrics.rie import calculate_rie
from scripts.performance.metrics.roce import calculate_roce
from scripts.performance.metrics.ckc import calculate_ckc

# Import other metrics as implemented


def calculate_metrics(
    scores: np.ndarray, labels: np.ndarray, percentile: float | list[float], metrics: list[str] | None = None
) -> dict[float, dict[str, float]]:
    """
    Calculate multiple virtual screening metrics.

    Parameters
    ----------
    scores : np.ndarray
        Screening scores
    labels : np.ndarray
        Binary labels (0/1)
    percentile : Union[float, List[float]]
        Percentile threshold(s) for selection
    metrics : List[str], optional
        Metrics to calculate. If None, calculates all.
        Available metrics:
        Threshold-dependent:
            - 'pm': Power Metric
            - 'ef': Enrichment Factor
            - 'ref': Relative Enrichment Factor
            - 'roce': ROC Enrichment
            - 'ccr': Correct Classification Rate
            - 'mcc': Matthews Correlation Coefficient
        Threshold-independent:
            - 'auc_roc': Area Under ROC Curve
            - 'aupr': Area Under Precision-Recall Curve
            - 'bedroc': Boltzmann-Enhanced Discrimination ROC
            - 'rie': Robust Initial Enhancement

    Returns:
    -------
    Dict[float, Dict[str, float]]
        Results dictionary organized by threshold
    """
    # Separate metrics into threshold-dependent and independent
    threshold_dependent = {
        "pm": calculate_power_metric,
        "ef": calculate_enrichment_factor,
        "ref": calculate_relative_enrichment_factor,
        "roce": calculate_roce,
        "ccr": calculate_ccr,
        "mcc": calculate_mcc,
        "ef_alt": calculate_enrichment_factor_alt,
        "rdkit_ef": calculate_rdkit_ef,
        "ckc": calculate_ckc,
    }

    threshold_independent = {
        "auc_roc": calculate_auc_roc,
        "aupr": calculate_aupr,
        "bedroc": calculate_bedroc,
        "rie": calculate_rie,
        "rdkit_auc": calculate_rdkitauc,
    }

    all_metrics = {**threshold_dependent, **threshold_independent}

    # Validate metrics
    metrics = metrics or list(all_metrics.keys())
    invalid_metrics = set(metrics) - set(all_metrics.keys())
    if invalid_metrics:
        raise ValueError(f"Invalid metrics requested: {invalid_metrics}")

    # Initialize results
    percentiles = np.atleast_1d(percentile)
    results = {p: {} for p in percentiles}

    # Calculate threshold-dependent metrics
    dependent_metrics = [m for m in metrics if m in threshold_dependent]
    for metric in dependent_metrics:
        metric_values = threshold_dependent[metric](scores, labels, percentiles)
        metric_values = np.atleast_1d(metric_values)
        for p, value in zip(percentiles, metric_values, strict=False):
            results[p][metric] = value

    # Calculate threshold-independent metrics
    independent_metrics = [m for m in metrics if m in threshold_independent]
    for metric in independent_metrics:
        # These metrics return a single value regardless of threshold
        value = threshold_independent[metric](scores, labels)
        # Add same value to all threshold results
        for p in percentiles:
            results[p][metric] = value

    return results
