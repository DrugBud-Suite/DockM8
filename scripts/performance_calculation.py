import itertools
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit.ML.Scoring import Scoring
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Search for 'DockM8' in parent directories
dockm8_path = next((p / 'DockM8' for p in Path(__file__).resolve().parents if (p / 'DockM8').is_dir()), None)
sys.path.append(str(dockm8_path))

from scripts.consensus_methods import CONSENSUS_METHODS
from scripts.postprocessing import rank_scores, standardize_scores
from scripts.utilities.utilities import parallel_executor, printlog

warnings.filterwarnings("ignore")

def process_combination(combination, clustering_method, ranked_df, standardised_df, actives_df, percentages):
    filtered_ranked_df = ranked_df[['ID'] + list(combination)]
    filtered_standardised_df = standardised_df[['ID'] + list(combination)]
    combination_dfs = []
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
        merged_df.fillna(0, inplace=True)
        scores = merged_df[col_to_sort].values
        activities = merged_df['Activity'].values
        # Calculate EF for all percentages at once using vectorized operations
        auc_roc = round(roc_auc_score(activities, scores, multi_class='ovo'), 3)
        bedroc = round(Scoring.CalcBEDROC(list(zip(scores, activities)), 1, 80.5), 3)
        auc = round(Scoring.CalcAUC(list(zip(scores, activities)), 1), 3)
        ef_results = [calculate_EF(merged_df, p) for p in percentages]
        rie = round(Scoring.CalcRIE(list(zip(scores, activities)), 1, 80.5), 3)
        combination_dfs.append(pd.DataFrame({'clustering': clustering_method,
                                        'consensus': method,
                                        'scoring': '_'.join(list(combination)),
                                        'AUC_ROC': auc_roc,
                                        'BEDROC': bedroc,
                                        'AUC': auc,
                                        **{f'EF_{p}%': ef for p, ef in zip(percentages, ef_results)},
                                        'RIE': rie}, index=[0]))
    combination_df = pd.concat(combination_dfs, axis=0)
    return combination_df

def calculate_performance_for_clustering_method(dir, w_dir, actives_df, percentages):
        clustering_method = '_'.join(dir.split('_')[1:3]) if len(dir.split('_')) > 3 else dir.split('_')[1] if len(dir.split('_')) == 3 else None
        # Calculate performance for single scoring functions
        rescored_df = pd.read_csv(Path(w_dir) / dir / 'allposes_rescored.csv')
        standardised_df = standardize_scores(rescored_df, 'min_max')
        standardised_df['ID'] = standardised_df['Pose ID'].str.split('_').str[0]
        standardised_df['ID'] = standardised_df['ID'].astype(str)
        score_columns = [col for col in standardised_df.columns if col not in ['Pose ID', 'ID']]
        result_list = []
        merged_df = pd.merge(standardised_df, actives_df, on='ID')
        merged_df.fillna(0, inplace=True)
        for col in score_columns:
            merged_df.sort_values(col, ascending=False, inplace=True)
            scores = merged_df[col].values
            activities = merged_df['Activity'].values
            # Calculate EF for all percentages at once using vectorized operations
            auc_roc = round(roc_auc_score(activities, scores, multi_class='ovo'), 3)
            bedroc = round(Scoring.CalcBEDROC(list(zip(scores, activities)), 1, 80.5), 3)
            auc = round(Scoring.CalcAUC(list(zip(scores, activities)), 1), 3)
            ef_results = [calculate_EF(merged_df, p) for p in percentages]
            rie = round(Scoring.CalcRIE(list(zip(scores, activities)), 1, 80.5), 3)
            result_list.append(pd.DataFrame({
                'clustering': clustering_method,
                'consensus': 'None',
                'scoring': col,
                'AUC_ROC': auc_roc,
                'BEDROC': bedroc,
                'AUC': auc,
                **{f'EF_{p}%': ef for p, ef in zip(percentages, ef_results)},
                'RIE': rie}, index=[0]))
        # Calculate performance for consensus scoring functions
        ranked_df = rank_scores(standardised_df)
        ranked_df['ID'] = ranked_df['Pose ID'].str.split('_').str[0]
        ranked_df['ID'] = ranked_df['ID'].astype(str)
        for length in tqdm(range(2, len(score_columns)), desc=f'{clustering_method}'):
            combinations = list(itertools.combinations(score_columns, length))
            # For each combination
            results = parallel_executor(process_combination, combinations, ncpus = 50, backend = 'concurrent_process_silent', clustering_method = clustering_method, ranked_df = ranked_df, standardised_df = standardised_df, actives_df = actives_df, percentages = percentages)
            for result in results:
                result_list.append(result)
        return pd.concat(result_list, axis=0)

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
    dirs = [dir for dir in os.listdir(w_dir) if dir.startswith('rescoring') and dir.endswith('clustered')]
    results = parallel_executor(calculate_performance_for_clustering_method, dirs, ncpus = math.ceil(len(dirs)//2), backend = 'concurrent_process_silent', w_dir=w_dir, actives_df = actives_df, percentages=percentages)
    all_results = pd.concat(results, ignore_index=True)
    (w_dir / 'performance').mkdir(parents=True, exist_ok=True)
    all_results.to_csv(Path(w_dir) / "performance" / 'performance.csv', index=False)
    return all_results

def calculate_EF(merged_df, percentage: float):
    total_rows = len(merged_df)
    N100_percent = total_rows

    Nx_percent = round((percentage / 100) * total_rows)
    Hits100_percent = np.sum(merged_df['Activity'])

    Hitsx_percent = np.sum(merged_df.head(Nx_percent)['Activity'])

    ef = (Hitsx_percent / Nx_percent) * (N100_percent / Hits100_percent)
    if ef > 100:
        return 100
    else:
        return round(ef, 2)
