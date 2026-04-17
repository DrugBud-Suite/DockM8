#!/usr/bin/env python3
"""
Optimized consensus analyzer implementation using Numba for improved performance.
"""

import time
import sys
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from importlib import import_module
import numba
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Ensure paths are correctly set up for importing consensus methods
scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents
                    if (p / "scripts").is_dir()), None)
if scripts_path and scripts_path.exists():
    sys.path.append(str(scripts_path.parent))

# Optimized RESCORING_FUNCTIONS - only storing what's needed
RESCORING_FUNCTIONS = {
    # "AAScore": {"best_value": "min"},  # Removed from publication
    "AD4": {"best_value": "min"},
    "CHEMPLP": {"best_value": "min"},
    "ConvexPLR": {"best_value": "max"},
    "GNINA-Affinity": {"best_value": "min"},
    "CNN-Score": {"best_value": "max"},
    "CNN-Affinity": {"best_value": "max"},
    "GenScore-scoring": {"best_value": "max"},
    "GenScore-docking": {"best_value": "max"},
    "GenScore-balanced": {"best_value": "max"},
    "KORP-PL": {"best_value": "min"},
    "LinF9": {"best_value": "min"},
    "NNScore": {"best_value": "max"},
    # "PLECScore": {"best_value": "max"},  # Removed from publication
    "PLP": {"best_value": "min"},
    "RFScoreVS": {"best_value": "max"},
    "RTMScore": {"best_value": "max"},
    "SCORCH": {"best_value": "max"},
    "Vinardo": {"best_value": "min"}
}

# Numba-optimized utility functions
@numba.jit(nopython=True)
def normalize_scores_numba(scores, best_value_is_min):
    """
    Normalize scores to [0,1] range where 1 is always best (Numba optimized).
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.full_like(scores, 0.5)
        
    normalized = (scores - min_score) / (max_score - min_score)
    if best_value_is_min:
        normalized = 1 - normalized
    return normalized

@numba.jit(nopython=True)
def prepare_screening_data_numba(scores, labels, percentile):
    """
    Prepare screening data by sorting and applying percentile threshold.
    """
    # Sort by descending scores
    indices = np.argsort(scores)[::-1]
    sorted_scores = scores[indices]
    sorted_labels = labels[indices]
    
    # Calculate number of compounds to select
    n_selected = int(np.ceil(len(scores) * percentile / 100))
    
    return sorted_labels, sorted_scores, n_selected

# Numba-optimized metric calculation functions
@numba.jit(nopython=True)
def calculate_enrichment_factor_numba(sorted_labels, n_selected, n_total, n_actives_total):
    """Calculate Enrichment Factor using Numba."""
    ns = np.sum(sorted_labels[:n_selected])  # Number of actives in selection
    ef = (n_total * ns) / (n_actives_total * n_selected)
    return ef

@numba.jit(nopython=True)
def calculate_relative_enrichment_factor_numba(sorted_labels, n_selected, n_actives_total):
    """Calculate Relative Enrichment Factor using Numba."""
    ns = np.sum(sorted_labels[:n_selected])  # Number of actives in selection
    max_possible_actives = min(n_selected, n_actives_total)
    ref = (100 * ns) / max_possible_actives
    return ref

@numba.jit(nopython=True)
def calculate_roce_numba(sorted_labels, n_selected, n_total, n_actives_total):
    """Calculate ROC Enrichment using Numba."""
    ns = np.sum(sorted_labels[:n_selected])  # Number of actives in selection
    denominator = n_actives_total * (n_selected - ns)
    if denominator == 0:
        return np.inf
    roce = (ns * (n_total - n_actives_total)) / denominator
    return roce

@numba.jit(nopython=True)
def calculate_power_metric_numba(sorted_labels, n_selected, n_total, n_actives_total):
    """Calculate Power Metric using Numba."""
    ns = np.sum(sorted_labels[:n_selected])  # Number of actives in selection
    numerator = ns * n_total - n_actives_total * ns
    denominator = ns * n_total - 2 * n_actives_total * ns + n_actives_total * n_selected
    if denominator == 0:
        return 0
    pm = numerator / denominator
    return pm

@numba.jit(nopython=True)
def calculate_mcc_numba(sorted_labels, n_selected, n_total, n_actives_total):
    """Calculate Matthews Correlation Coefficient using Numba."""
    ns = np.sum(sorted_labels[:n_selected])  # Number of actives in selection
    numerator = n_total * ns - n_selected * n_actives_total
    denominator = np.sqrt(n_selected * n_actives_total * (n_total - n_actives_total) * (n_total - n_selected))
    if denominator == 0:
        return 0
    mcc = numerator / denominator
    return mcc

@numba.jit(nopython=True)
def calculate_ccr_numba(sorted_labels, n_selected, n_total, n_actives_total):
    """Calculate Correct Classification Rate using Numba."""
    ns = np.sum(sorted_labels[:n_selected])  # Number of actives in selection
    if n_actives_total == 0 or n_total - n_actives_total == 0:
        return 0
    sensitivity = ns / n_actives_total
    specificity = (n_total - n_selected - (n_actives_total - ns)) / (n_total - n_actives_total)
    ccr = 0.5 * (sensitivity + specificity)
    return ccr

@numba.jit(nopython=True)
def calculate_ckc_numba(sorted_labels, n_selected, n_total, n_actives_total):
    """Calculate Cohen's Kappa Coefficient (CKC) using Numba."""
    ns = np.sum(sorted_labels[:n_selected])  # Number of actives in selection
    
    numerator = n_total * n_actives_total + n_total * n_selected - 2 * ns * n_total
    denominator = n_total * n_actives_total + n_total * n_selected - 2 * n_actives_total * n_selected
    
    # Calculate Cohen's Kappa
    if denominator == 0:
        return 0
    else:
        ckc = 1 - (numerator / denominator)
    
    return ckc

# Consensus methods optimized for Numba
@numba.jit(nopython=True)
def ecr_consensus_numba(values, n_samples):
    """Calculate the ECR consensus score using Numba."""
    sigma = 0.05 * n_samples
    
    # Vectorized rank calculation
    ranks = np.zeros_like(values, dtype=np.float64)
    for i in range(values.shape[1]):
        col = values[:, i]
        # Create sorting indices and assign ranks in descending order
        idx = np.argsort(-col)
        for j in range(len(idx)):
            ranks[idx[j], i] = j + 1
            
    ecr_scores = np.exp(-ranks / sigma)
    scores = np.zeros(values.shape[0])
    for i in range(values.shape[0]):
        scores[i] = np.sum(ecr_scores[i, :]) / sigma
    return scores

@numba.jit(nopython=True)
def rbr_consensus_numba(values, n_samples):
    """Calculate RBR score using Numba."""
    n_cols = values.shape[1]
    ranks = np.zeros_like(values, dtype=np.float64)
    
    for i in range(n_cols):
        col = values[:, i]
        idx = np.argsort(-col)
        for j in range(len(idx)):
            ranks[idx[j], i] = j + 1
    
    # Calculate mean ranks manually for each row
    mean_ranks = np.zeros(values.shape[0])
    for i in range(values.shape[0]):
        mean_ranks[i] = np.sum(ranks[i, :]) / n_cols
    
    # Invert ranks so higher scores are better
    max_rank = n_samples
    inverted_ranks = max_rank - mean_ranks + 1
    return inverted_ranks

@numba.jit(nopython=True)
def zscore_consensus_numba(values):
    """Calculate Z-score using Numba."""
    # Calculate means and standard deviations for each column
    means = np.zeros(values.shape[1])
    stds = np.zeros(values.shape[1])
    
    for i in range(values.shape[1]):
        means[i] = np.mean(values[:, i])
        stds[i] = np.std(values[:, i])
        # Handle zero standard deviation
        if stds[i] == 0:
            stds[i] = 1.0
            
    # Calculate z-scores
    z_scores = np.zeros_like(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            z_scores[i, j] = (values[i, j] - means[j]) / stds[j]
    
    # Calculate row means
    result = np.zeros(values.shape[0])
    for i in range(values.shape[0]):
        result[i] = np.mean(z_scores[i, :])
    
    return result

# Function to calculate BEDROC
def calculate_bedroc_wrapper(scores, labels, alpha=80.5):
    """
    Calculate BEDROC score using the exact same method as the original implementation.
    
    This directly mirrors the original implementation in bedroc.py.
    """
    # Import RDKit's BEDROC calculation function
    from rdkit.ML.Scoring.Scoring import CalcBEDROC
    
    # Create scoring data in RDKit's expected format
    # Following the exact pattern from the original implementation
    rdkit_data = []
    # Sort by scores in descending order first (to exactly match original)
    sorted_indices = scores.argsort()[::-1]
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Create the data pairs
    for s, l in zip(sorted_scores, sorted_labels, strict=False):
        rdkit_data.append([float(s), int(l)])
    
    # Call RDKit's BEDROC function with the same parameters
    bedroc_value = CalcBEDROC(rdkit_data, 1, alpha)
    return round(bedroc_value, 3)

# Unified metrics calculation function
def calculate_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    percentile: float | list[float],
    metrics: list[str] | None = None
) -> dict[float, dict[str, float]]:
    """
    Calculate multiple virtual screening metrics with Numba optimization.
    """
    # Convert percentiles to numpy array
    percentiles = np.atleast_1d(percentile)
    
    # Validate inputs
    for p in percentiles:
        if p <= 0 or p > 100:
            raise ValueError("Percentile thresholds must be between 0 and 100")
    
    # Define threshold-dependent metrics with Numba optimization
    threshold_dependent = {
        "pm": calculate_power_metric_numba,
        "ef": calculate_enrichment_factor_numba,
        "ref": calculate_relative_enrichment_factor_numba,
        #"roce": calculate_roce_numba,
        #"ccr": calculate_ccr_numba,
        #"mcc": calculate_mcc_numba,
        #"ckc": calculate_ckc_numba,
    }

    # Define threshold-independent metrics
    threshold_independent = {
        "auc_roc": lambda s, l: round(roc_auc_score(l, s), 3),
        #"aupr": lambda s, l: round(average_precision_score(l, s), 3),
        "bedroc": calculate_bedroc_wrapper
    }

    all_metrics = {**threshold_dependent, **threshold_independent}

    # Validate metrics
    metrics = metrics or list(all_metrics.keys())
    invalid_metrics = set(metrics) - set(all_metrics.keys())
    if invalid_metrics:
        raise ValueError(f"Invalid metrics requested: {invalid_metrics}")

    # Initialize results
    results = {p: {} for p in percentiles}

    n_total = len(scores)
    n_actives_total = np.sum(labels)

    # Pre-calculate threshold-independent metrics
    independent_metrics = [m for m in metrics if m in threshold_independent]
    for metric in independent_metrics:
        value = threshold_independent[metric](scores, labels)
        # Add same value to all threshold results
        for p in percentiles:
            results[p][metric] = value

    # Calculate threshold-dependent metrics
    dependent_metrics = [m for m in metrics if m in threshold_dependent]
    for p in percentiles:
        sorted_labels, sorted_scores, n_selected = prepare_screening_data_numba(scores, labels, p)
        
        for metric in dependent_metrics:
            if metric == "ref":
                # Special case for REF which doesn't need n_total
                value = threshold_dependent[metric](sorted_labels, n_selected, n_actives_total)
            else:
                value = threshold_dependent[metric](sorted_labels, n_selected, n_total, n_actives_total)
            results[p][metric] = round(float(value), 3)

    return results

# Helper function for batching combinations
def batch_combinations(combinations_list, batch_size=10):
    """Split combinations into balanced batches for better parallel processing."""
    # Sort combinations by length to balance workload (longer combinations take more time)
    sorted_combinations = sorted(combinations_list, key=len, reverse=True)
    
    # Create balanced batches using a simple greedy algorithm
    batches = []
    current_batch = []
    
    for combo in sorted_combinations:
        current_batch.append(combo)
        if len(current_batch) >= batch_size:
            batches.append(current_batch)
            current_batch = []
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

class ConsensusAnalyzer:
    def __init__(
        self,
        scoring_data_path: Path | str,
        activity_data_path: Path | str,
        id_column: str = "ID",
        activity_column: str = "Activity",
        thresholds: list[float] = [1, 2, 5],
        n_jobs: int = -1,
        batch_size: int = 10
    ):
        """
        Initialize the consensus analysis system with support for single function analysis.
        
        Parameters
        ----------
        scoring_data_path : Path | str
            Path to CSV file containing scoring results
        activity_data_path : Path | str
            Path to CSV file containing activity labels (0/1)
        id_column : str, optional
            Name of the ID column in both files
        activity_column : str, optional
            Name of the activity column in activity data
        thresholds : List[float], optional
            Thresholds for metric calculation (in percentages)
        n_jobs : int, optional
            Number of parallel jobs (-1 for all cores)
        batch_size : int, optional
            Size of combination batches for processing
        """
        self.id_column = id_column
        self.activity_column = activity_column
        self.thresholds = thresholds
        self.n_jobs = n_jobs if n_jobs > 0 else None
        self.batch_size = batch_size
        
        print("Loading and validating data...")
        self.scoring_data, self.activity_data = self._load_and_validate_data(
            scoring_data_path, activity_data_path
        )
        
        print("Normalizing scoring functions...")
        self._normalize_scoring_data()

    def _load_and_validate_data(
        self, scoring_path: Path | str, activity_path: Path | str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate input data files."""
        # Load data using pandas
        scoring_df = pd.read_csv(scoring_path)
        activity_df = pd.read_csv(activity_path)
        
        # Check if data is empty
        if scoring_df.empty or activity_df.empty:
            raise ValueError("Empty scoring or activity file")
        
        # Validate scoring data columns
        if self.id_column not in scoring_df.columns:
            raise ValueError(f"Column '{self.id_column}' not found in scoring data")
        
        # Validate activity data columns
        if self.id_column not in activity_df.columns:
            raise ValueError(f"Column '{self.id_column}' not found in activity data")
        if self.activity_column not in activity_df.columns:
            raise ValueError(f"Column '{self.activity_column}' not found in activity data")
        
        # Identify scoring functions
        self.available_functions = [
            col for col in scoring_df.columns
            if col in RESCORING_FUNCTIONS
        ]
        
        if not self.available_functions:
            raise ValueError("No recognized scoring functions found in data")
            
        print(f"Found {len(self.available_functions)} scoring functions: "
            f"{', '.join(self.available_functions)}")
            
        # Check for missing values
        missing_values = scoring_df[self.available_functions].isnull().any().any()
        
        if missing_values:
            raise ValueError("Scoring data contains missing values in scoring function columns")
            
        if activity_df[self.activity_column].isnull().any():
            raise ValueError("Activity data contains missing values in activity column")
        
        # Check if activity values are binary (0 or 1)
        unique_activities = activity_df[self.activity_column].unique()
        
        if not set(unique_activities).issubset({0, 1}):
            raise ValueError("Activity values must be binary (0 or 1)")
        
        # Get common IDs
        common_ids = set(scoring_df[self.id_column]) & set(activity_df[self.id_column])
        
        if len(common_ids) == 0:
            raise ValueError("No common IDs found between scoring and activity data")
        
        # Filter data to only include common IDs
        scoring_df = scoring_df[scoring_df[self.id_column].isin(common_ids)]
        activity_df = activity_df[activity_df[self.id_column].isin(common_ids)]
        
        print(f"Found {len(common_ids)} common IDs between datasets")
        
        return scoring_df, activity_df
    
    def _normalize_scoring_data(self):
        """Normalize all scoring function data using Numba."""
        # Create normalized data dictionary with IDs
        self.normalized_data = {}
        self.normalized_data[self.id_column] = self.scoring_data[self.id_column].values
        
        # Process each scoring function
        for sf_name in tqdm(self.available_functions, desc="Normalizing scoring functions"):
            scores = self.scoring_data[sf_name].values
            best_value_is_min = RESCORING_FUNCTIONS[sf_name]["best_value"] == "min"
            
            # Use Numba for normalization
            normalized = normalize_scores_numba(scores, best_value_is_min)
            self.normalized_data[sf_name] = normalized
    
    def _process_single_function(self, function_name: str) -> list[dict]:
        """Process a single scoring function."""
        try:
            # Get scores
            scores = self.normalized_data[function_name]
            
            # Get activity labels for matching IDs
            activity_dict = {
                id_val: act for id_val, act in zip(
                    self.activity_data[self.id_column],
                    self.activity_data[self.activity_column], strict=False
                )
            }
            
            # Align scores with activity labels
            aligned_scores = []
            aligned_labels = []
            
            for idx, id_val in enumerate(self.normalized_data[self.id_column]):
                if id_val in activity_dict:
                    aligned_scores.append(scores[idx])
                    aligned_labels.append(activity_dict[id_val])
            
            # Convert to NumPy arrays
            aligned_scores = np.array(aligned_scores)
            aligned_labels = np.array(aligned_labels)
            
            # Calculate metrics
            metrics = calculate_metrics(
                scores=aligned_scores,
                labels=aligned_labels,
                percentile=self.thresholds
            )
            
            # Record results
            results = []
            for threshold, threshold_metrics in metrics.items():
                results.append({
                    "scoring_function": function_name,
                    "threshold": threshold,
                    "n_compounds": len(aligned_scores),
                    **threshold_metrics
                })
            
            return results
            
        except Exception as e:
            return [{
                "scoring_function": function_name,
                "error": str(e)
            }]
            
    def _process_batch_combinations(self, combinations_batch, consensus_methods):
        """Process a batch of combinations in a single worker."""
        all_results = []
        failed_tasks = []
        
        for combination in combinations_batch:
            try:
                results = self._process_combination(combination, consensus_methods)
                if results and "error" in results[0]:
                    failed_tasks.append(("+".join(combination), results[0]["error"]))
                else:
                    all_results.extend(results)
            except Exception as e:
                failed_tasks.append(("+".join(combination), str(e)))
        
        return all_results, failed_tasks
    
    def _process_batch_single_functions(self, functions_batch):
        """Process a batch of single scoring functions."""
        all_results = []
        failed_tasks = []
        
        for function_name in functions_batch:
            try:
                results = self._process_single_function(function_name)
                if results and "error" in results[0]:
                    failed_tasks.append((function_name, results[0]["error"]))
                else:
                    all_results.extend(results)
            except Exception as e:
                failed_tasks.append((function_name, str(e)))
        
        return all_results, failed_tasks

    def _apply_consensus_method(self, data, method_name, ids):
        """
        Apply a consensus method to the data.
        
        For RBV and SoftRBV, we use the original implementation by 
        converting data to DataFrame format first.
        """
        try:
            n_samples = data.shape[0]
            
            # Create a temporary DataFrame for original implementations
            if method_name in ["rbv", "soft_rbv"]:
                # Convert NumPy array back to DataFrame for original implementation
                columns = [f"col_{i}" for i in range(data.shape[1])]
                df = pd.DataFrame(data, columns=columns)
                df[self.id_column] = ids
                
                # Import original implementations dynamically
                if method_name == "rbv":
                    try:
                        # First try direct import
                        from scripts.consensus.methods.rbv_tiebreaker import rbv_consensus
                    except ImportError:
                        # If that fails, use importlib
                        rbv_module = import_module('scripts.consensus.methods.rbv_tiebreaker')
                        rbv_consensus = rbv_module.rbv_consensus
                        
                    result_df = rbv_consensus(df, columns, self.id_column)
                    # Extract scores and ensure they're in the same order as input
                    result_dict = dict(zip(result_df[self.id_column], result_df["RbV"], strict=False))
                    return np.array([result_dict.get(id_val, 0) for id_val in ids])
                
                elif method_name == "soft_rbv":
                    try:
                        # First try direct import
                        from scripts.consensus.methods.soft_rbv import soft_rbv_consensus
                    except ImportError:
                        # If that fails, use importlib
                        soft_rbv_module = import_module('scripts.consensus.methods.soft_rbv')
                        soft_rbv_consensus = soft_rbv_module.soft_rbv_consensus
                        
                    result_df = soft_rbv_consensus(df, columns, self.id_column)
                    # Extract scores and ensure they're in the same order as input
                    result_dict = dict(zip(result_df[self.id_column], result_df["SoftRbV"], strict=False))
                    return np.array([result_dict.get(id_val, 0) for id_val in ids])
            
            # Use optimized Numba functions for other methods
            elif method_name == "ecr":
                return ecr_consensus_numba(data, n_samples)
            elif method_name == "rbr":
                return rbr_consensus_numba(data, n_samples)
            elif method_name == "zscore":
                return zscore_consensus_numba(data)
            else:
                raise ValueError(f"Unknown consensus method: {method_name}")
                
        except Exception as e:
            print(f"Error applying consensus method {method_name}: {str(e)}")
            # Return default scores (zeros) if method fails
            return np.zeros(data.shape[0])
            
    def _process_combination(
        self,
        combination: tuple[str, ...],
        consensus_methods: list[str]
    ) -> list[dict]:
        """Process a combination of scoring functions."""
        try:
            # Extract data for this combination
            data = np.column_stack([self.normalized_data[col] for col in combination])
            ids = self.normalized_data[self.id_column]
            
            # Get activity labels
            activity_dict = {
                id_val: act for id_val, act in zip(
                    self.activity_data[self.id_column],
                    self.activity_data[self.activity_column], strict=False
                )
            }
            
            results = []
            
            for method in consensus_methods:
                try:
                    # Apply consensus method
                    consensus_scores = self._apply_consensus_method(data, method, ids)
                    
                    # Skip if method failed and returned zeros
                    if np.all(consensus_scores == 0):
                        continue
                    
                    # Align scores with activity labels
                    aligned_scores = []
                    aligned_labels = []
                    
                    for idx, id_val in enumerate(ids):
                        if id_val in activity_dict:
                            aligned_scores.append(consensus_scores[idx])
                            aligned_labels.append(activity_dict[id_val])
                    
                    # Convert to NumPy arrays
                    aligned_scores = np.array(aligned_scores)
                    aligned_labels = np.array(aligned_labels)
                    
                    # Calculate metrics
                    metrics = calculate_metrics(
                        scores=aligned_scores,
                        labels=aligned_labels,
                        percentile=self.thresholds
                    )
                    
                    # Record results
                    for threshold, threshold_metrics in metrics.items():
                        results.append({
                            "combination": "+".join(combination),
                            "consensus_method": method,
                            "threshold": threshold,
                            "n_compounds": len(aligned_scores),
                            **threshold_metrics
                        })
                except Exception as e:
                    print(f"Error processing combination {'+'.join(combination)} with method {method}: {str(e)}")
            
            return results
            
        except Exception as e:
            return [{
                "combination": "+".join(combination),
                "error": str(e)
            }]

    def analyze(
        self,
        output_path: Path | str | None = None,
        include_single: bool = True
    ) -> pd.DataFrame:
        """
        Perform complete consensus analysis with optional single function analysis.
        
        Parameters
        ----------
        output_path : Optional[Path | str]
            Path to save results CSV file
        include_single : bool
            Whether to include single function analysis
        
        Returns:
        -------
        pd.DataFrame
            Analysis results
        """
        start_time = time.time()
        results = []
        failed_tasks = []
        
        # Process single functions if requested
        if include_single:
            print("\nAnalyzing individual scoring functions...")
            # Process single functions in batches
            batch_size = min(self.batch_size, len(self.available_functions))
            func_batches = [self.available_functions[i:i+batch_size]
                          for i in range(0, len(self.available_functions), batch_size)]
            
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for batch in func_batches:
                    future = executor.submit(self._process_batch_single_functions, batch)
                    futures.append(future)
                
                for future in tqdm(as_completed(futures), total=len(futures),
                                 desc="Processing single functions"):
                    try:
                        batch_results, batch_failed = future.result()
                        results.extend(batch_results)
                        failed_tasks.extend(batch_failed)
                    except Exception as e:
                        print(f"Error in single function batch: {str(e)}")
        
        # Generate all possible combinations
        print("\nGenerating scoring function combinations...")
        combinations_list = []
        for r in range(2, len(self.available_functions) + 1):
            combinations_list.extend(combinations(self.available_functions, r))
        
        print(f"Generated {len(combinations_list)} combinations")
        
        # Create balanced batches of combinations for better load distribution
        combo_batches = batch_combinations(combinations_list, self.batch_size)
        print(f"Split into {len(combo_batches)} balanced batches")
        
        # Process combinations in balanced batches
        print("\nAnalyzing scoring function combinations...")
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for batch in combo_batches:
                future = executor.submit(
                    self._process_batch_combinations,
                    batch,
                    ["ecr", "rbr", "zscore", "rbv", "soft_rbv"]  # List consensus methods directly
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures),
                             desc="Processing combination batches", smoothing=0):
                try:
                    batch_results, batch_failed = future.result()
                    results.extend(batch_results)
                    failed_tasks.extend(batch_failed)
                except Exception as e:
                    print(f"Error in combination batch: {str(e)}")
        
        # Combine results efficiently
        print("\nAssembling final results...")
        final_df = pd.DataFrame(results) if results else pd.DataFrame()
        
        if failed_tasks:
            print("\nFailed tasks:")
            for task, error in failed_tasks[:10]:  # Show only first 10 failures to avoid overwhelming output
                print(f"{task}: {error}")
            if len(failed_tasks) > 10:
                print(f"...and {len(failed_tasks) - 10} more failures")
        
        if output_path and not final_df.empty:
            print(f"\nSaving results to {output_path}")
            final_df.to_csv(output_path, index=False)
        
        print(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds")
        
        return final_df


def run_consensus_analysis(
    scoring_data_path: Path | str,
    activity_data_path: Path | str,
    output_path: Path | str | None = None,
    thresholds: list[float] = [1, 2, 5],
    id_column: str = "ID",
    activity_column: str = "Activity",
    n_jobs: int = -1,
    include_single: bool = True,
    batch_size: int = 10
) -> pd.DataFrame:
    """
    Convenience function to run consensus analysis.
    
    Parameters
    ----------
    scoring_data_path : Path | str
        Path to CSV file containing scoring results
    activity_data_path : Path | str
        Path to CSV file containing activity labels
    output_path : Optional[Path | str]
        Path to save results CSV file
    thresholds : List[float]
        Thresholds for metric calculation (percentages)
    id_column : str
        Name of the ID column
    activity_column : str
        Name of the activity column
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    include_single : bool
        Whether to include single function analysis
    batch_size : int
        Size of combination batches for processing
    
    Returns:
    -------
    pd.DataFrame
        Analysis results
    """
    analyzer = ConsensusAnalyzer(
        scoring_data_path=scoring_data_path,
        activity_data_path=activity_data_path,
        thresholds=thresholds,
        id_column=id_column,
        activity_column=activity_column,
        n_jobs=n_jobs,
        batch_size=batch_size
    )
    
    return analyzer.analyze(output_path=output_path, include_single=include_single)
