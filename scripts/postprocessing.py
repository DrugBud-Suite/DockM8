from rdkit.Chem import PandasTools
import pandas as pd
import os
from scripts.utilities import *
from scripts.consensus_methods import *
from pathlib import Path
import itertools
from joblib import Parallel, delayed
import json
from scripts.rescoring_functions import RESCORING_FUNCTIONS
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
    def min_max_standardization(score, best_value):
        """
        Perform min-max standardization on the given score.
        """
        if best_value == 'max':
            standardized_scores = (score - score.min()) / (score.max() - score.min())
        else:
            standardized_scores = (score.max() - score) / (score.max() - score.min())
        return standardized_scores
    def min_max_standardization_scaled(score, min_value, max_value):
        """
        Performs min-max standardization scaling on a given score.
        """
        standardized_scores = (score - min_value) / (max_value - min_value)
        return standardized_scores
    if standardization_type == 'min_max':
        # Standardize each score column using min-max standardization
        for col in df.columns:
            if col != 'Pose ID':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = min_max_standardization(df[col], RESCORING_FUNCTIONS[col][2])
    elif standardization_type == 'scaled':
        # Standardize each score column using scaled min-max standardization
        for col in df.columns:
            if col != 'Pose ID':
                column_info = RESCORING_FUNCTIONS[col]
                if column_info:
                    col_min = column_info[3]
                    col_max = column_info[4]
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = min_max_standardization_scaled(df[col], col_min, col_max)
    elif standardization_type == 'percentiles':
        # Standardize each score column using min-max standardization based on percentiles
        for col in df.columns:
            if col != 'Pose ID':
                column_info = RESCORING_FUNCTIONS[col]
                if column_info:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    column_data = df[col].dropna().values
                    col_max = np.percentile(column_data, 99)
                    col_min = np.percentile(column_data, 1)
                    df[col] = min_max_standardization_scaled(df[col], RESCORING_FUNCTIONS[col][2], col_min, col_max)
    else:
        raise ValueError(f"Invalid standardization type: {standardization_type}")
    return df


def rank_scores(df):
    """
    Ranks the scores in the given DataFrame in descending order for each column, excluding 'Pose ID' and 'ID'.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the scores to be ranked.

    Returns:
        pandas.DataFrame: The DataFrame with the scores ranked in descending order for each column.
    """
    df = df.assign(**{col: df[col].rank(method='average', ascending=False) for col in df.columns if col not in ['Pose ID', 'ID']})
    return df

def apply_consensus_methods(w_dir : str, clustering_metric : str, method : str, rescoring_functions : list, standardization_type : str):
    """
    Applies consensus methods to rescored data and saves the results to a CSV file.

    Args:
    w_dir (str): The working directory where the rescored data is located.
    clustering_metric (str): The clustering metric used to cluster the poses.
    method (str): The consensus method to apply.
    rescoring_functions (list): A list of rescoring functions to apply.
    standardization_type (str): The type of standardization to apply to the scores.

    Returns:
    None
    """
    # Create the 'ranking' directory if it doesn't exist
    (Path(w_dir) / 'ranking').mkdir(parents=True, exist_ok=True)
    # Read the rescored data from the CSV file
    rescoring_folder = f'rescoring_{clustering_metric}_clustered'
    rescored_dataframe = pd.read_csv(Path(w_dir) / rescoring_folder / 'allposes_rescored.csv')
    # Standardize the scores and add the 'ID' column
    standardized_dataframe = standardize_scores(rescored_dataframe, standardization_type)
    standardized_dataframe['ID'] = standardized_dataframe['Pose ID'].str.split('_').str[0]
    # Rank the scores and add the 'ID' column
    ranked_dataframe = rank_scores(standardized_dataframe)
    ranked_dataframe['ID'] = ranked_dataframe['Pose ID'].str.split('_').str[0]
    # Create the 'consensus' directory if it doesn't exist
    (Path(w_dir) / 'consensus').mkdir(parents=True, exist_ok=True)
    # Define the rank and score methods
    rank_methods = {'method1': method1_ECR_best,
                    'method2': method2_ECR_average,
                    'method3': method3_avg_ECR,
                    'method4': method4_RbR}
    score_methods = {'method5': method5_RbV,
                    'method6': method6_Zscore_best,
                    'method7': method7_Zscore_avg}
    # Apply the selected consensus method to the data
    if method in rank_methods:
        method_function = rank_methods[method]
        consensus_dataframe = method_function(ranked_dataframe, clustering_metric, [col for col in ranked_dataframe.columns if col not in ['Pose ID', 'ID']])
    elif method in score_methods:
        method_function = score_methods[method]
        consensus_dataframe = method_function(standardized_dataframe, clustering_metric, [col for col in standardized_dataframe.columns if col not in ['Pose ID', 'ID']])
    else:
        raise ValueError(f"Invalid consensus method: {method}")
    # Drop the 'Pose ID' column and save the consensus results to a CSV file
    consensus_dataframe = consensus_dataframe.drop(columns="Pose ID", errors='ignore')
    consensus_dataframe.to_csv(Path(w_dir) / 'consensus' / f'{clustering_metric}_{method}_results.csv', index=False)
    return

def ensemble_consensus(receptors:list, clustering_metric : str, method : str, threshold : float or int):
    """
    Given a list of receptor file paths, this function reads the consensus clustering results for each receptor,
    selects the top n compounds based on a given threshold, and returns a list of common compounds across all receptors.
    
    Parameters:
    -----------
    receptors : list of str
        List of file paths to receptor files.
    clustering_metric : str
        The clustering metric used to generate the consensus clustering results.
    method : str
        The clustering method used to generate the consensus clustering results.
    threshold : float or int
        The percentage of top compounds to select from each consensus clustering result.
    
    Returns:
    --------
    list of str
        List of common compounds across all receptors.
    """
    topn_dataframes = []
    # Iterate over each receptor file
    for receptor in receptors:
        w_dir = Path(receptor).parent / Path(receptor).stem
        # Read the consensus clustering results for the receptor
        consensus_file = pd.read_csv(Path(w_dir) / 'consensus' / f'{clustering_metric}_{method}_results.csv')
        # Select the top n compounds based on the given threshold
        consensus_file_topn = consensus_file.head(math.ceil(consensus_file.shape[0] * (threshold/100)))
        # Append the top n compounds dataframe to the list
        topn_dataframes.append(consensus_file_topn)
    # Find the common compounds across all receptors
    common_compounds = set(topn_dataframes[0]['ID'])
    # Find the intersection of 'ID' values with other dataframes
    for df in topn_dataframes[1:]:
        common_compounds.intersection_update(df['ID'])
    common_compounds_list = list(common_compounds)
    # Create a dataframe with the common compounds and save it to a CSV file
    common_compounds_df = pd.DataFrame(common_compounds_list, columns=['Common Compounds'])
    common_compounds_df.to_csv(Path(receptors[0].parent / 'ensemble_results.csv'), index=False)
    return list(common_compounds)