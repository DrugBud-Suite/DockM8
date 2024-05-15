import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools

# Search for 'DockM8' in parent directories
scripts_path = next((p / 'scripts' for p in Path(__file__).resolve().parents if (p / 'scripts').is_dir()), None)
dockm8_path = scripts_path.parent
sys.path.append(str(dockm8_path))

from scripts.consensus_methods import CONSENSUS_METHODS
from scripts.rescoring_functions import RESCORING_FUNCTIONS
from scripts.utilities.utilities import printlog

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def standardize_scores(df : pd.DataFrame, standardization_type : str):
    """
    Standardizes the scores in the given dataframe.

    Args:
    - df: pandas dataframe containing the scores to be standardized
    - standardization_type: string indicating the type of standardization ('min_max', 'scaled', 'percentiles')

    Returns:
    - df: pandas dataframe with standardized scores
    """
    def min_max_standardization(score, best_value, min_value, max_value):
        """
        Performs min-max standardization scaling on a given score using the defined min and max values.
        """
        return (score - min_value) / (max_value - min_value) if best_value == 'max' else (max_value - score) / (max_value - min_value)

    for col in df.columns:
        if col not in ['Pose ID', 'ID']:
            # Convert column to numeric values
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Get information about the scoring function
            function_info = RESCORING_FUNCTIONS.get(col)
            if function_info:
                if standardization_type == 'min_max':
                    # Standardise using the score's (current distribution) min and max values
                    df[col] = min_max_standardization(df[col], function_info['best_value'], df[col].min(), df[col].max())
                elif standardization_type == 'scaled':
                    # Standardise using the range defined in the RESCORING_FUNCTIONS dictionary
                    df[col] = min_max_standardization(df[col], function_info['best_value'], *function_info['range'])
                elif standardization_type == 'percentiles':
                    # Standardise using the 1st and 99th percentiles of this distribution
                    column_data = df[col].dropna().values
                    col_min, col_max = np.percentile(column_data, [1, 99]) if function_info['best_value'] == 'max' else np.percentile(column_data, [99, 1])
                    df[col] = min_max_standardization(df[col], function_info['best_value'], col_min, col_max)
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

def apply_consensus_methods(w_dir : str, selection_method : str, consensus_methods : str, rescoring_functions : list, standardization_type : str):
    """
    Applies consensus methods to rescored data and saves the results to a CSV file.

    Args:
    w_dir (str): The working directory where the rescored data is located.
    selection_method (str): The clustering metric used to cluster the poses.
    consensus_methods (str): The consensus methods to apply.
    rescoring_functions (list): A list of rescoring functions to apply.
    standardization_type (str): The type of standardization to apply to the scores.

    Returns:
    None
    """
    # Check if consensus_methods is None or 'None'
    if consensus_methods is None or consensus_methods == 'None':
        return printlog('No consensus methods selected, skipping consensus.')
    else:
        printlog(f"Applying consensus methods: {consensus_methods}")
        # Create the 'ranking' directory if it doesn't exist
        (Path(w_dir) / 'ranking').mkdir(parents=True, exist_ok=True)
        # Read the rescored data from the CSV file
        rescoring_folder = f'rescoring_{selection_method}_clustered'
        rescored_dataframe = pd.read_csv(Path(w_dir) / rescoring_folder / 'allposes_rescored.csv')
        # Standardize the scores and add the 'ID' column
        standardized_dataframe = standardize_scores(rescored_dataframe, standardization_type)
        standardized_dataframe['ID'] = standardized_dataframe['Pose ID'].str.split('_').str[0]
        # Rank the scores and add the 'ID' column
        ranked_dataframe = rank_scores(standardized_dataframe)
        ranked_dataframe['ID'] = ranked_dataframe['Pose ID'].str.split('_').str[0]
        # Ensure consensus_methods is a list even if it's a single string
        if isinstance(consensus_methods, str):
            consensus_methods = [consensus_methods]
        for consensus_method in consensus_methods:
            # Create the 'consensus' directory if it doesn't exist
            (Path(w_dir) / 'consensus').mkdir(parents=True, exist_ok=True)
            # Check if consensus_method is valid
            if consensus_method not in CONSENSUS_METHODS:
                raise ValueError(f"Invalid consensus method: {consensus_method}")
            # Get the method information from the dictionary
            conensus_info = CONSENSUS_METHODS[consensus_method]
            conensus_type = conensus_info['type']
            conensus_function = conensus_info['function']
            # Apply the selected consensus method to the data
            if conensus_type == 'rank':
                consensus_dataframe = conensus_function(ranked_dataframe, selection_method, [col for col in ranked_dataframe.columns if col not in ['Pose ID', 'ID']])
            elif conensus_type == 'score':
                consensus_dataframe = conensus_function(standardized_dataframe, selection_method, [col for col in standardized_dataframe.columns if col not in ['Pose ID', 'ID']])
            else:
                raise ValueError(f"Invalid consensus method type: {conensus_type}")
            # Drop the 'Pose ID' column and save the consensus results to a CSV file
            consensus_dataframe = consensus_dataframe.drop(columns="Pose ID", errors='ignore')
            consensus_dataframe = consensus_dataframe.sort_values(by='ID')
            # Save the consensus results to a CSV file or SDF file depending on the selection method
            if selection_method in ['bestpose_GNINA', 'bestpose_SMINA', 'bestpose_PLANTS', 'bestpose_QVINAW', 'bestpose_QVINA2']+list(RESCORING_FUNCTIONS.keys()):
                poses = PandasTools.LoadSDF(str(w_dir / 'clustering' / f'{selection_method}_clustered.sdf'), molColName='Molecule', idName='Pose ID')
                poses['ID'] = poses['Pose ID'].str.split('_').str[0]
                poses = poses[['ID', 'Molecule']]
                consensus_dataframe = pd.merge(consensus_dataframe, poses, on='ID', how='left')
                PandasTools.WriteSDF(consensus_dataframe, str(w_dir / 'consensus' / f'{selection_method}_{consensus_method}_results.sdf'), molColName='Molecule', idName='ID', properties=list(consensus_dataframe.columns))
            else:
                consensus_dataframe.to_csv(Path(w_dir) / 'consensus' / f'{selection_method}_{consensus_method}_results.csv', index=False)
        return

def ensemble_consensus(receptors:list, selection_method : str, consensus_method : str, threshold : float):
    """
    Given a list of receptor file paths, this function reads the consensus clustering results for each receptor,
    selects the top n compounds based on a given threshold, and returns a list of common compounds across all receptors.
    
    Parameters:
    -----------
    receptors : list of str
        List of file paths to receptor files.
    selection_method : str
        The clustering metric used to generate the consensus clustering results.
    consensus_method : str
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
        if selection_method in ['bestpose_GNINA', 'bestpose_SMINA', 'bestpose_PLANTS', 'bestpose_QVINAW', 'bestpose_QVINA2']+list(RESCORING_FUNCTIONS.keys()):
            consensus_file = PandasTools.LoadSDF(str(w_dir / 'consensus' / f'{selection_method}_{consensus_method}_results.sdf'), molColName='Molecule', idName='ID')
        else:
            consensus_file = pd.read_csv(Path(w_dir) / 'consensus' / f'{selection_method}_{consensus_method}_results.csv')
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
    
    common_compounds_df = pd.DataFrame()
    
    for receptor in receptors:
        w_dir = Path(receptor).parent / Path(receptor).stem
        # Read the consensus clustering results for the receptor
        if selection_method in ['bestpose_GNINA', 'bestpose_SMINA', 'bestpose_PLANTS', 'bestpose_QVINAW', 'bestpose_QVINA2']+list(RESCORING_FUNCTIONS.keys()):
            consensus_file = PandasTools.LoadSDF(str(w_dir / 'consensus' / f'{selection_method}_{consensus_method}_results.sdf'), molColName='Molecule', idName='ID')
        else:
            consensus_file = pd.read_csv(Path(w_dir) / 'consensus' / f'{selection_method}_{consensus_method}_results.csv')
        consensus_file = consensus_file[consensus_file['ID'].isin(common_compounds_list)]
        consensus_file['Receptor'] = Path(receptor).stem
        common_compounds_df = pd.concat([common_compounds_df, consensus_file], axis=0)
    # Save the common compounds and CSV or SDF file
    if selection_method in ['bestpose_GNINA', 'bestpose_SMINA', 'bestpose_PLANTS', 'bestpose_QVINAW', 'bestpose_QVINA2']+list(RESCORING_FUNCTIONS.keys()):
        PandasTools.WriteSDF(common_compounds_df, str(Path(receptors[0]).parent / 'ensemble_results.sdf'), molColName='Molecule', idName='ID', properties=list(common_compounds_df.columns))
    else:
        common_compounds_df.to_csv(Path(receptors[0]).parent / 'ensemble_results.csv', index=False)
    return common_compounds_df