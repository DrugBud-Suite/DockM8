import itertools
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools

from scripts.consensus_methods import (
    CONSENSUS_METHODS,
)
from scripts.postprocessing import rank_scores, standardize_scores
from scripts.utilities import parallel_executor, printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def calculate_performance(w_dir : Path, actives_library : Path, percentages : list):
    """
    Calculates the performance of a scoring system for different clustering methods.

    Args:
        w_dir (Path): The directory path where the scoring results are stored.
        actives_library (Path): The path to the actives library in SDF format.
        percentages (list): A list of percentages for calculating the EF (enrichment factor).

    Returns:
        DataFrame: A DataFrame containing the performance results for different clustering methods, consensus methods, and scoring functions.
    """
    printlog('Calculating performance...')
    #all_results = pd.DataFrame(columns=['clustering', 'consensus', 'scoring'] + [f'EF{p}' for p in percentages])
    #Load actives data
    actives_df = PandasTools.LoadSDF(str(actives_library), molColName=None, idName='ID')
    actives_df = actives_df[['ID', 'Activity']]
    actives_df['Activity'] = pd.to_numeric(actives_df['Activity'])
    # Calculate performance for each clustering method
    dirs = [dir for dir in os.listdir(w_dir) if dir.startswith('rescoring')]
    
    global calculate_performance_for_clustering_method
    def calculate_performance_for_clustering_method(dir):
        all_results = pd.DataFrame(columns=['clustering', 'consensus', 'scoring'] + [f'EF{p}' for p in percentages])
        clustering_method = '_'.join(dir.split('_')[1:3]) if len(dir.split('_')) > 3 else dir.split('_')[1] if len(dir.split('_')) == 3 else None
        # Calculate performance for single scoring functions
        rescored_df = pd.read_csv(Path(w_dir) / dir / 'allposes_rescored.csv')
        standardised_df = standardize_scores(rescored_df, 'min_max')
        standardised_df['ID'] = standardised_df['Pose ID'].str.split('_').str[0]
        standardised_df['ID'] = standardised_df['ID'].astype(str)
        score_columns = [col for col in standardised_df.columns if col not in ['Pose ID', 'ID']]
        for col in score_columns:
            filtered_df = standardised_df[['ID', col]]
            merged_df = pd.merge(filtered_df, actives_df, on='ID')
            merged_df.sort_values(col, ascending=False, inplace=True)
            ef_results = {}
            for p in percentages:
                ef = calculate_EF(merged_df, p)
                ef_results[f'EF{p}'] = ef
            all_results.loc[len(all_results)] = [clustering_method, 'None', col] + list(ef_results.values())
        # Calculate performance for consensus scoring functions
        ranked_df = rank_scores(standardised_df)
        ranked_df['ID'] = ranked_df['Pose ID'].str.split('_').str[0]
        ranked_df['ID'] = ranked_df['ID'].astype(str)
        global process_combination
        def process_combination(combination, clustering_method, ranked_df, standardised_df, actives_df, percentages):
            filtered_ranked_df = ranked_df[['ID'] + list(combination)]
            filtered_standardised_df = standardised_df[['ID'] + list(combination)]
            combination_df = pd.DataFrame()
            consensus_dfs = {}
            # For each consensus method
            for method in CONSENSUS_METHODS.keys():
                if CONSENSUS_METHODS[method]['type'] == 'rank':
                    consensus_dfs[method] = CONSENSUS_METHODS[method]['function'](filtered_ranked_df, clustering_method, list(combination))
                elif CONSENSUS_METHODS[method]['type'] == 'score':
                    consensus_dfs[method] = CONSENSUS_METHODS[method]['function'](filtered_standardised_df, clustering_method, list(combination))
                merged_df = pd.merge(consensus_dfs[method], actives_df, on='ID')
                # Get the column name that is not 'ID' or 'Activity'
                col_to_sort = [col for col in merged_df.columns if col not in ['ID', 'Activity']][0]
                merged_df.sort_values(col_to_sort, ascending=False, inplace=True)
                ef_results = np.zeros((len(percentages),), dtype=np.float64)
                for i, p in enumerate(percentages):
                    ef = calculate_EF(merged_df, p)
                    ef_results[i] = ef
                method_result = [clustering_method, method, '_'.join(list(combination))] + list(ef_results)
                method_result_df = pd.DataFrame([method_result], columns=['clustering', 'consensus', 'scoring'] + [f'EF{p}' for p in percentages])
                combination_df = pd.concat([combination_df, method_result_df], axis=0)
            return combination_df
        # For any length of combination of scoring functions
        result_list = []
        for length in (range(2, len(score_columns))):
            combinations = list(itertools.combinations(score_columns, length))
            # For each combination
            results = parallel_executor(process_combination, combinations, ncpus = 10, backend = 'concurrent_process_silent', clustering_method = clustering_method, ranked_df = ranked_df, standardised_df = standardised_df, actives_df = actives_df, percentages = percentages)
            for result in results:
                result_list.append(result)
        all_results = pd.concat(result_list, axis=0)
        return all_results
    res = parallel_executor(calculate_performance_for_clustering_method, dirs, ncpus = math.ceil(os.cpu_count()/len(dirs)), backend = 'concurrent_process_silent')
    all_results = pd.concat(res, ignore_index=True)
    (w_dir / 'performance').mkdir(parents=True, exist_ok=True)
    all_results.to_csv(Path(w_dir) / "performance" / 'performance.csv', index=False)
    return all_results


def calculate_EF(merged_df, percentage : float):
    """
    Calculates the Enrichment Factor (EF) for a given percentage.

    Args:
        merged_df (DataFrame): The merged DataFrame containing the scores and activity data.
        percentage (int): The percentage for calculating the EF.

    Returns:
        float: The calculated EF.
    """
    Nx_percent = round((percentage/100) * len(merged_df))
    N100_percent = len(merged_df)
    Hitsx_percent = merged_df.head(Nx_percent)['Activity'].sum()
    Hits100_percent = merged_df['Activity'].sum()
    ef = round((Hitsx_percent / Nx_percent) * (N100_percent / Hits100_percent), 2)
    return ef

