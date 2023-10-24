from rdkit.Chem import PandasTools
import pandas as pd
import os
from scripts.utilities import *
from scripts.consensus_methods import *
from pathlib import Path
import itertools
from joblib import Parallel, delayed
import json

import pandas as pd

def standardize_scores(df : pd.DataFrame, standardization_type : str):
    """
    Standardizes the scores in the given dataframe.

    Args:
    - df: pandas dataframe containing the scores to be standardized
    - standardization_type: string indicating the type of standardization ('min_max', 'scaled', 'percentiles')

    Returns:
    - df: pandas dataframe with standardized scores
    """
    def min_max_standardisation(score, best_value):
        if best_value == 'max':
            standardized_scores = (score - score.min()) / (score.max() - score.min())
        else:
            standardized_scores = (score.max() - score) / (score.max() - score.min())
        return standardized_scores

    def min_max_standardization_scaled(score, min_value, max_value):
        standardized_scores = (score - min_value) / (max_value - min_value)
        return standardized_scores

    if standardization_type == 'min_max':
        rescoring_functions_standardization = {'GNINA_Affinity': 'min',
                                                'CNN-Score': 'max',
                                                'CNN-Affinity': 'max',
                                                'Vinardo': 'min',
                                                'AD4': 'min',
                                                'LinF9': 'min',
                                                'RFScoreVS': 'max',
                                                'PLP': 'min',
                                                'CHEMPLP': 'min',
                                                'NNScore': 'max',
                                                'PLECnn': 'max',
                                                'AAScore': 'min',
                                                'ECIF': 'max',
                                                'SCORCH': 'max',
                                                'RTMScore': 'max',
                                                'KORPL': 'min',
                                                'ConvexPLR': 'max'}
        for col in df.columns:
            if col != 'Pose ID':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = min_max_standardisation(df[col], rescoring_functions_standardization[col])

    elif standardization_type == 'scaled':
        with open('rescoring_functions.json', 'r') as json_file:
            rescoring_functions = json.load(json_file)
        for col in df.columns:
            if col != 'Pose ID':
                column_info = rescoring_functions.get(col)
                if column_info:
                    col_min = column_info['parameters']['min']
                    col_max = column_info['parameters']['max']
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = min_max_standardization_scaled(df[col], col_min, col_max)

    elif standardization_type == 'percentiles':
        with open('rescoring_functions.json', 'r') as json_file:
            rescoring_functions = json.load(json_file)
        for col in df.columns:
            if col != 'Pose ID':
                column_info = rescoring_functions.get(col)
                if column_info:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    column_data = df[col].dropna().values  # Drop NaN values and convert to numpy array
                    col_max = np.percentile(column_data, 99)
                    col_min = np.percentile(column_data, 1)
                    df[col] = min_max_standardisation(df[col], column_info['parameters']['order'], col_min, col_max)

    else:
        raise ValueError(f"Invalid standardization type: {standardization_type}")

    return df

def rank_scores(df):
    df = df.assign(**{col: df[col].rank(method='average', ascending=False) for col in df.columns if col not in ['Pose ID', 'ID']})
    return df

def apply_consensus_methods(w_dir : str, clustering_metric : str, method : str, rescoring_functions : list, standardization_type : str):
    """
    Applies consensus methods to a set of poses and saves the results in CSV files.

    Args:
    - w_dir (str): path to the working directory
    - clustering_metric (str): the clustering metric to use
    - method (str): the consensus method to use
    - rescoring_functions (list): a list of rescoring functions to use
    - standardization_type (str): the type of standardization to use

    Returns:
    - None
    """
    (Path(w_dir) / 'ranking').mkdir(parents=True, exist_ok=True)
    
    rescoring_folder = f'rescoring_{clustering_metric}_clustered'
    rescored_dataframe = pd.read_csv(Path(w_dir) / rescoring_folder / 'allposes_rescored.csv')

    # Standardize and rank the scores
    standardized_dataframe = standardize_scores(rescored_dataframe, standardization_type)
    ranked_dataframe = rank_scores(standardized_dataframe)

    # Extract ID from 'Pose ID' and save to CSV
    ranked_dataframe['ID'] = ranked_dataframe['Pose ID'].str.split('_').str[0]

    (Path(w_dir) / 'consensus').mkdir(parents=True, exist_ok=True)
    
    rank_methods = {'method1': method1_ECR_best,
                    'method2': method2_ECR_average,
                    'method3': method3_avg_ECR,
                    'method4': method4_RbR}
    score_methods = {'method5': method5_RbV,
                    'method6': method6_Zscore_best,
                    'method7': method7_Zscore_avg}

    if method in rank_methods:
        method_function = rank_methods[method]
        consensus_dataframe = method_function(ranked_dataframe, clustering_metric, [col for col in ranked_dataframe.columns if col not in ['Pose ID', 'ID']])
    elif method in score_methods:
        method_function = score_methods[method]
        consensus_dataframe = method_function(standardized_dataframe, clustering_metric, [col for col in standardized_dataframe.columns if col not in ['Pose ID', 'ID']])
    else:
        raise ValueError(f"Invalid consensus method: {method}")


    consensus_dataframe = consensus_dataframe.drop(columns="Pose ID", errors='ignore')
    consensus_dataframe.to_csv(Path(w_dir) / 'consensus' / f'{clustering_metric}_{method}_results.csv', index=False)
    return


