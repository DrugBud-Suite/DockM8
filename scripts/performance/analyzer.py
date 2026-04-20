import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

scripts_path = next((p / "scripts" for p in Path(__file__).resolve().parents if (p / "scripts").is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.consensus.consensus import _METHODS
from scripts.performance.calculate_metrics import calculate_metrics
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS


def normalize_scores(scores: np.ndarray, scoring_info: dict) -> np.ndarray:
    """
    Normalize scores to [0,1] range where 1 is always best.
    Optimized implementation that handles edge cases efficiently.
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.full_like(scores, 0.5)
        
    normalized = (scores - min_score) / (max_score - min_score)
    if scoring_info["best_value"] == "min":
        normalized = 1 - normalized
    return normalized


def normalize_scores_batch(data_array: np.ndarray, scoring_infos: list[dict]) -> np.ndarray:
    """Vectorized normalization of multiple scoring functions at once."""
    result = np.empty_like(data_array)
    
    for i, info in enumerate(scoring_infos):
        column_data = data_array[:, i]
        result[:, i] = normalize_scores(column_data, info)
        
    return result


def batch_combinations(combinations_list, batch_size=100):
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
        batch_size: int = 100
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
        
        # Pre-merge the activity data with normalized scores once
        self.merged_data = pd.merge(
            self.normalized_data,
            self.activity_data[[self.id_column, self.activity_column]],
            on=self.id_column,
            how='inner'
        )
        
        # Prepare arrays for faster access in metrics calculation
        self.labels_array = self.merged_data[self.activity_column].values
        
        # Cache frequently accessed calculated values
        self.n_total = len(self.merged_data)
        self.n_actives_total = np.sum(self.labels_array)
        
        # Pre-compute indices mapping for efficient slicing
        self._prepare_indices_mapping()

    def _prepare_indices_mapping(self):
        """
        Pre-compute indices mapping for efficient data extraction
        during metrics calculation.
        """
        self.function_to_idx = {func: i for i, func in enumerate(self.available_functions)}
        
    def _load_and_validate_data(
        self, scoring_path: Path | str, activity_path: Path | str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate input data files."""
        scoring_df = pd.read_csv(scoring_path)
        activity_df = pd.read_csv(activity_path)
        
        # Check if data is empty first
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
        
        # Identify scoring functions before missing value check
        self.available_functions = [
            col for col in scoring_df.columns
            if col in RESCORING_FUNCTIONS
        ]
        
        if not self.available_functions:
            raise ValueError("No recognized scoring functions found in data")
            
        print(f"Found {len(self.available_functions)} scoring functions: "
            f"{', '.join(self.available_functions)}")
        
        # Check for missing values only in scoring function columns and activity column
        if scoring_df[self.available_functions].isnull().any().any():
            raise ValueError("Scoring data contains missing values in scoring function columns")
        if activity_df[self.activity_column].isnull().any():
            raise ValueError("Activity data contains missing values in activity column")
        
        # Validate activity values efficiently
        activity_values = pd.unique(activity_df[self.activity_column])
        if not set(activity_values).issubset({0, 1}):
            raise ValueError("Activity values must be binary (0 or 1)")
        
        # Verify common IDs using efficient set operations
        common_ids = set(scoring_df[self.id_column]) & set(activity_df[self.id_column])
        if not common_ids:
            # Only raise error if datasets are not empty
            if not scoring_df.empty and not activity_df.empty:
                raise ValueError("No common IDs found between scoring and activity data")
        else:
            # Filter data to only include common IDs for better efficiency
            scoring_df = scoring_df[scoring_df[self.id_column].isin(common_ids)]
            activity_df = activity_df[activity_df[self.id_column].isin(common_ids)]
            print(f"Found {len(common_ids)} common IDs between datasets")
        
        return scoring_df, activity_df
    
    def _normalize_scoring_data(self):
        """Normalize all scoring function data using vectorized operations."""
        # Pre-allocate the normalized DataFrame with optimal memory usage
        self.normalized_data = pd.DataFrame({self.id_column: self.scoring_data[self.id_column]})
        
        # Extract scoring function data as numpy array for faster operations
        data_array = self.scoring_data[self.available_functions].values
        scoring_infos = [RESCORING_FUNCTIONS[sf] for sf in self.available_functions]
        
        # Use vectorized normalization for better performance
        normalized_array = normalize_scores_batch(data_array, scoring_infos)
        
        # Assign normalized data back to DataFrame
        for i, sf_name in enumerate(self.available_functions):
            self.normalized_data[sf_name] = normalized_array[:, i]

    def _process_batch_combinations(self, combinations_batch, consensus_methods):
        """
        Process a batch of combinations in a single worker.
        This reduces overhead of process spawning.
        """
        results = []
        failed = []
        
        # Pre-compute method names for faster lookup
        method_names = list(_METHODS.keys())
        
        for combination in combinations_batch:
            try:
                # Extract only the needed columns for this combination
                combination_data = self.merged_data[[self.id_column, self.activity_column] + list(combination)]
                
                for method_idx, method in enumerate(consensus_methods):
                    # Apply consensus method directly
                    consensus_result = method(
                        combination_data,
                        list(combination),
                        id_column=self.id_column
                    )
                    
                    # Get consensus score column (always the last column)
                    score_col = consensus_result.columns[-1]
                    
                    # Get scores as numpy array for faster processing
                    consensus_scores = consensus_result[score_col].values
                    
                    # Merge consensus scores with original IDs and labels for metrics calculation
                    # We'll use numpy arrays directly instead of DataFrame merges
                    id_to_idx = {id_val: i for i, id_val in enumerate(consensus_result[self.id_column])}
                    score_indices = [id_to_idx.get(id_val) for id_val in combination_data[self.id_column]]
                    
                    # Get ordered scores and corresponding labels
                    ordered_scores = np.array([consensus_scores[idx] if idx is not None else np.nan for idx in score_indices])
                    valid_mask = ~np.isnan(ordered_scores)
                    final_scores = ordered_scores[valid_mask]
                    final_labels = combination_data[self.activity_column].values[valid_mask]
                    
                    # Calculate metrics directly
                    metrics = calculate_metrics(
                        scores=final_scores,
                        labels=final_labels,
                        percentile=self.thresholds
                    )
                    
                    # Record results
                    for threshold, threshold_metrics in metrics.items():
                        results.append({
                            "combination": "+".join(combination),
                            "consensus_method": method_names[method_idx],
                            "threshold": threshold,
                            "n_compounds": len(final_scores),
                            **threshold_metrics
                        })
            except Exception as e:
                failed.append(("+".join(combination), str(e)))
        
        return pd.DataFrame(results) if results else pd.DataFrame(), failed
    
    def _process_batch_single_functions(self, functions_batch):
        """Process a batch of single scoring functions."""
        results = []
        failed = []
        
        for function_name in functions_batch:
            try:
                # Extract scores for this function
                scores = self.merged_data[function_name].values
                labels = self.labels_array
                
                # Calculate metrics directly
                metrics = calculate_metrics(
                    scores=scores,
                    labels=labels,
                    percentile=self.thresholds
                )
                
                # Record results
                for threshold, threshold_metrics in metrics.items():
                    results.append({
                        "scoring_function": function_name,
                        "threshold": threshold,
                        "n_compounds": len(scores),
                        **threshold_metrics
                    })
            except Exception as e:
                failed.append((function_name, str(e)))
        
        return pd.DataFrame(results) if results else pd.DataFrame(), failed
    
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
            # Process single functions in batches for better efficiency
            func_batches = [self.available_functions[i:i+self.batch_size]
                          for i in range(0, len(self.available_functions), self.batch_size)]
            
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(self._process_batch_single_functions, batch): batch
                    for batch in func_batches
                }
                
                for future in tqdm(as_completed(futures), total=len(func_batches),
                                 desc="Processing single functions"):
                    try:
                        result_df, failed = future.result()
                        if not result_df.empty:
                            results.append(result_df)
                        failed_tasks.extend(failed)
                    except Exception as e:
                        batch = futures[future]
                        failed_tasks.extend([(func, str(e)) for func in batch])
        
        # Generate all possible combinations more efficiently
        print("\nGenerating scoring function combinations...")
        combinations_list = []
        for r in range(2, len(self.available_functions) + 1):
            combinations_list.extend(combinations(self.available_functions, r))
        
        print(f"Generated {len(combinations_list)} combinations")
        
        # Create balanced batches of combinations for better load distribution
        combo_batches = batch_combinations(combinations_list, self.batch_size)
        print(f"Split into {len(combo_batches)} balanced batches")
        
        # Get consensus methods directly as functions
        consensus_methods = list(_METHODS.values())
        
        # Process combinations in balanced batches
        print("\nAnalyzing scoring function combinations...")
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(self._process_batch_combinations, batch, consensus_methods): batch
                for batch in combo_batches
            }
            
            for future in tqdm(as_completed(futures), total=len(combo_batches),
                             desc="Processing combination batches", smoothing=0):
                try:
                    result_df, failed = future.result()
                    if not result_df.empty:
                        results.append(result_df)
                    failed_tasks.extend(failed)
                except Exception as e:
                    batch = futures[future]
                    failed_tasks.extend([
                        ("+".join(combo), str(e))
                        for combo in batch
                    ])
        
        # Combine results efficiently
        print("\nAssembling final results...")
        final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        
        if failed_tasks:
            print("\nFailed tasks:")
            for task, error in failed_tasks:
                print(f"{task}: {error}")
        
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
    batch_size: int = 100
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
