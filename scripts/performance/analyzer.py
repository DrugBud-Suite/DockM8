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

from scripts.consensus.consensus import _METHODS, apply_consensus_scoring
from scripts.performance.calculate_metrics import calculate_metrics
from scripts.rescoring.rescoring import RESCORING_FUNCTIONS


def normalize_scores(scores: np.ndarray, scoring_info: dict) -> np.ndarray:
    """
    Normalize scores to [0,1] range where 1 is always best.
    
    Parameters
    ----------
    scores : np.ndarray
        Raw scores to normalize
    scoring_info : Dict
        Scoring function information from RESCORING_FUNCTIONS
        
    Returns:
    -------
    np.ndarray
        Normalized scores in [0,1] range
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    if max_score == min_score:
        return np.full_like(scores, 0.5)
        
    normalized = (scores - min_score) / (max_score - min_score)
    if scoring_info["best_value"] == "min":
        normalized = 1 - normalized
    return normalized

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
        """Normalize all scoring function data efficiently."""
        # Pre-allocate the normalized DataFrame
        self.normalized_data = pd.DataFrame({self.id_column: self.scoring_data[self.id_column]})
        
        # Vectorized normalization for all scoring functions
        for sf_name in tqdm(self.available_functions, desc="Normalizing scoring functions"):
            scores = self.scoring_data[sf_name].values
            self.normalized_data[sf_name] = normalize_scores(scores, RESCORING_FUNCTIONS[sf_name])

    def _process_single_function(self, function_name: str) -> pd.DataFrame:
        """Process a single scoring function."""
        try:
            # Get scores and merge with activity data
            scores_df = self.normalized_data[[self.id_column, function_name]]
            merged_data = pd.merge(
                scores_df,
                self.activity_data[[self.id_column, self.activity_column]],
                on=self.id_column,
                how='inner'
            )
            
            # Calculate metrics
            metrics = calculate_metrics(
                scores=merged_data[function_name].values,
                labels=merged_data[self.activity_column].values,
                percentile=self.thresholds
            )
            
            # Record results
            results = []
            for threshold, threshold_metrics in metrics.items():
                results.append({
                    "scoring_function": function_name,
                    "threshold": threshold,
                    "n_compounds": len(merged_data),
                    **threshold_metrics
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            return pd.DataFrame([{
                "scoring_function": function_name,
                "error": str(e)
            }])

    def _process_combination(
        self,
        combination: tuple[str, ...],
        consensus_methods: list[str]
    ) -> pd.DataFrame:
        """Process a combination of scoring functions."""
        try:
            consensus_data = self.normalized_data[[self.id_column] + list(combination)].copy()
            
            results = []
            for method in consensus_methods:
                consensus_result = apply_consensus_scoring(
                    data=consensus_data,
                    methods=[method],
                    id_column=self.id_column,
                    normalize=False
                )
                
                score_col = [col for col in consensus_result.columns
                           if col != self.id_column][0]
                
                merged_data = pd.merge(
                    consensus_result[[self.id_column, score_col]],
                    self.activity_data[[self.id_column, self.activity_column]],
                    on=self.id_column,
                    how='inner'
                )
                
                metrics = calculate_metrics(
                    scores=merged_data[score_col].values,
                    labels=merged_data[self.activity_column].values,
                    percentile=self.thresholds
                )
                
                for threshold, threshold_metrics in metrics.items():
                    results.append({
                        "combination": "+".join(combination),
                        "consensus_method": method,
                        "threshold": threshold,
                        "n_compounds": len(merged_data),
                        **threshold_metrics
                    })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            return pd.DataFrame([{
                "combination": "+".join(combination),
                "error": str(e)
            }])

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
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                future_to_function = {
                    executor.submit(self._process_single_function, func): func
                    for func in self.available_functions
                }
                
                for future in tqdm(
                    as_completed(future_to_function),
                    total=len(self.available_functions),
                    desc="Processing single functions"
                ):
                    function = future_to_function[future]
                    try:
                        result_df = future.result()
                        if "error" in result_df.columns:
                            failed_tasks.append((function, result_df.iloc[0]["error"]))
                        else:
                            results.append(result_df)
                    except Exception as e:
                        failed_tasks.append((function, str(e)))
        
        # Generate and process combinations
        print("\nAnalyzing scoring function combinations...")
        combinations_list = []
        for r in range(2, len(self.available_functions) + 1):
            combinations_list.extend(combinations(self.available_functions, r))
        
        print(f"Generated {len(combinations_list)} combinations")
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_combination = {
                executor.submit(
                    self._process_combination,
                    combination,
                    list(_METHODS.keys())
                ): combination
                for combination in combinations_list
            }
            
            for future in tqdm(
                as_completed(future_to_combination),
                total=len(combinations_list),
                desc="Processing combinations"
            ):
                combination = future_to_combination[future]
                try:
                    result_df = future.result()
                    if "error" in result_df.columns:
                        failed_tasks.append(
                            ("+".join(combination), result_df.iloc[0]["error"])
                        )
                    else:
                        results.append(result_df)
                except Exception as e:
                    failed_tasks.append(("+".join(combination), str(e)))
        
        # Combine and save results
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
    include_single: bool = True
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
        n_jobs=n_jobs
    )
    
    return analyzer.analyze(output_path=output_path, include_single=include_single)
