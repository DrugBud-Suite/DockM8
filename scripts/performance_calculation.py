from rdkit.Chem import PandasTools
import pandas as pd
import os
from scripts.utilities import *
from scripts.consensus_methods import *
from scripts.postprocessing import *
from pathlib import Path
import itertools
from joblib import Parallel, delayed
import json

import pandas as pd

def process_dataframes(w_dir, rescoring_folders):
    """
    Process the dataframes by reading CSV files, standardizing scores, and ranking scores.

    Args:
        w_dir (str): The working directory.
        rescoring_folders (dict): A dictionary containing the names of the rescoring folders.

    Returns:
        tuple: A tuple containing two dictionaries - standardised_dataframes and ranked_dataframes.
               standardised_dataframes: A dictionary containing the standardized dataframes.
               ranked_dataframes: A dictionary containing the ranked dataframes.
    """
    rescored_dataframes = {name: pd.read_csv(Path(w_dir) / rescoring_folders[name] / 'allposes_rescored.csv') for name in rescoring_folders}
    standardised_dataframes = {f'{name}_standardised': standardize_scores(rescored_dataframes[name], 'min_max') for name in rescoring_folders}
    ranked_dataframes = {f'{name}_ranked': rank_scores(standardised_dataframes[f'{name}_standardised']) for name in rescoring_folders}
    return standardised_dataframes, ranked_dataframes


def process_combination(combination, w_dir, name, standardised_df, ranked_df, column_mapping, rank_methods, score_methods, docking_library, original_df):
    """
    Process a combination of selected columns and calculate performance metrics.

    Args:
        combination (iterable): The combination of selected columns.
        w_dir (str): The working directory.
        name (str): The name of the combination.
        standardised_df (pandas.DataFrame): The standardized dataframe.
        ranked_df (pandas.DataFrame): The ranked dataframe.
        column_mapping (dict): The mapping of columns.
        rank_methods (dict): The dictionary of rank methods.
        score_methods (dict): The dictionary of score methods.
        docking_library (str): The docking library.
        original_df (pandas.DataFrame): The original dataframe.

    Returns:
        dict: A dictionary containing the performance metrics for each method.
    """
    selected_columns = list(combination)
    ranked_selected_columns = [column_mapping[col] for col in selected_columns]
    subset_name = '_'.join(selected_columns)
    replacements_dict = {'_R_': '', '_S_': '_'}
    for key, value in replacements_dict.items():
        subset_name = subset_name.replace(key, value)
        
    standardised_subset = standardised_df[['ID'] + selected_columns]
    ranked_subset = ranked_df[['ID'] + ranked_selected_columns]
    
    # Analyze the dataframes using rank methods and score methods
    analysed_dataframes = {method: rank_methods[method](ranked_subset, name, ranked_selected_columns) for method in rank_methods}
    analysed_dataframes.update({method: score_methods[method](standardised_subset, name, selected_columns) for method in score_methods})

    # Calculate the EF1% metric for a dataframe
    def calculate_EF1(df, w_dir, docking_library, original_df):
        """
        Calculate the EF1% metric for a dataframe.

        Args:
            df (pandas.DataFrame): The dataframe.
            w_dir (str): The working directory.
            docking_library (str): The docking library.
            original_df (pandas.DataFrame): The original dataframe.

        Returns:
            float: The EF1% metric.
        """
        merged_df = df.merge(original_df, on='ID')
        method_list = df.columns.tolist()[1:]
        method_ranking = {'ECR': False,
                            'Zscore': False,
                            'RbV': False,
                            'RbR': True}
        for method in method_list:
            asc = [method_ranking[key] for key in method_ranking if key in method][0]
            sorted_df = merged_df.sort_values(method, ascending=asc)
            N1_percent = round(0.01 * len(sorted_df))
            N100_percent = len(sorted_df)
            Hits1_percent = sorted_df.head(N1_percent)['Activity'].sum()
            Hits100_percent = sorted_df['Activity'].sum()
            ef1 = round((Hits1_percent / N1_percent) * (N100_percent / Hits100_percent), 2)
        return ef1
    
    result_dict = {}
    for method, df in analysed_dataframes.items():
        df = df.drop(columns="Pose ID", errors='ignore')
        enrichment_factor = calculate_EF1(df, w_dir, docking_library, original_df)
        ef_df = pd.DataFrame({'clustering_method': [name],
                                'method_name': [method],
                                'selected_columns': [subset_name],
                                'EF1%': [enrichment_factor]
        })

        result_dict[method] = ef_df
    return result_dict


def process_combination_wrapper(args):
    return process_combination(*args)

def apply_consensus_methods_combinations(w_dir, docking_library, clustering_metrics):
    """
    Apply consensus methods combinations to calculate performance metrics.

    Args:
        w_dir (str): The working directory path.
        docking_library (str): The path to the docking library.
        clustering_metrics (list): A list of clustering metrics.

    Returns:
        None
    """
    # Create 'ranking' directory if it doesn't exist
    (Path(w_dir) / 'ranking').mkdir(parents=True, exist_ok=True)
    # Create a dictionary of rescoring folders for each clustering metric
    rescoring_folders = {metric: f'rescoring_{metric}_clustered' for metric in clustering_metrics}
    # Process dataframes and get standardised and ranked dataframes
    standardised_dataframes, ranked_dataframes = process_dataframes(w_dir, rescoring_folders)
    # Save the dataframes to CSV files in the 'ranking' directory
    for name, df_dict in {'standardised': standardised_dataframes, 'ranked': ranked_dataframes}.items():
        for df_name, df in df_dict.items():
            df['ID'] = df['Pose ID'].str.split('_').str[0]
            df.to_csv(Path(w_dir) / 'ranking' / f'{df_name}.csv', index=False)
    # Create 'consensus' directory if it doesn't exist
    (Path(w_dir) / 'consensus').mkdir(parents=True, exist_ok=True)
    # Define rank methods and score methods
    rank_methods = {
        'method1': method1_ECR_best,
        'method2': method2_ECR_average,
        'method3': method3_avg_ECR,
        'method4': method4_RbR}
    score_methods = {
        'method5': method5_RbV,
        'method6': method6_Zscore_best,
        'method7': method7_Zscore_avg}
    # Load the docking library as a DataFrame
    original_df = PandasTools.LoadSDF(str(docking_library), molColName=None, idName='ID')
    original_df = original_df[['ID', 'Activity']]
    original_df['Activity'] = pd.to_numeric(original_df['Activity'])
    df_list = []
    printlog('Calculating consensus methods for every possible score combination...')
    # Iterate over each rescoring folder
    for name in tqdm(rescoring_folders, total=len(rescoring_folders)):
        standardised_df = standardised_dataframes[name + '_standardised']
        ranked_df = ranked_dataframes[name + '_ranked']
        calc_columns = [col for col in standardised_df.columns if col not in ['Pose ID', 'ID']]
        column_mapping = {col: f"{col}_R" for col in calc_columns}
        ranked_df = ranked_df.rename(columns=column_mapping)
        parallel = Parallel(n_jobs=int(os.cpu_count() - 2), backend='multiprocessing')
        # Generate combinations of score columns
        for L in range(2, len(calc_columns)):
            combinations = list(itertools.combinations(calc_columns, L))
            args = [
                (subset,
                 w_dir,
                 name,
                 standardised_df,
                 ranked_df,
                 column_mapping,
                 rank_methods,
                 score_methods,
                 docking_library,
                 original_df) for subset in combinations]
            results = parallel(delayed(process_combination_wrapper)(arg) for arg in args)
            for result_dict in results:
                for method, df in result_dict.items():
                    df_list.append(df)
            consensus_summary = pd.concat(df_list, ignore_index=True)
    # Save the consensus_summary DataFrame to a single CSV file
    consensus_summary = pd.concat(df_list, ignore_index=True)
    consensus_summary.to_csv(Path(w_dir) / 'consensus' / 'consensus_summary.csv', index=False)

def calculate_EF_single_functions(w_dir, docking_library, clustering_metrics):
    """
    Calculate EF (Enrichment Factor) for single scoring functions.

    Args:
        w_dir (str): The working directory path.
        docking_library (str): The path to the docking library.
        clustering_metrics (list): List of clustering metrics.

    Returns:
        None
    """
    # Create 'ranking' directory if it doesn't exist
    (Path(w_dir) / 'ranking').mkdir(parents=True, exist_ok=True)
    # Create a dictionary of rescoring folders for each clustering metric
    rescoring_folders = {metric: f'rescoring_{metric}_clustered' for metric in clustering_metrics}
    # Process dataframes and get standardised and ranked dataframes
    standardised_dataframes, ranked_dataframes = process_dataframes(w_dir, rescoring_folders)
    # Save standardised and ranked dataframes as CSV files
    for name, df_dict in {'standardised': standardised_dataframes, 'ranked': ranked_dataframes}.items():
        for df_name, df in df_dict.items():
            df['ID'] = df['Pose ID'].str.split('_').str[0]
            df.to_csv(Path(w_dir) / 'ranking' / f'{df_name}.csv',index=False)
    # Load the original docking library as a Pandas dataframe
    original_df = PandasTools.LoadSDF(str(docking_library), molColName=None, idName='ID')
    original_df = original_df[['ID', 'Activity']]
    original_df['Activity'] = pd.to_numeric(original_df['Activity'])
    # Create an empty dataframe to store EF results
    EF_results = pd.DataFrame(columns=['Scoring Function', 'Clustering Metric', 'EF10%', 'EF1%'])
    # Calculate EFs for separate scoring functions
    for file in os.listdir(Path(w_dir) / 'ranking'):
        if file.endswith('_standardised.csv'):
            clustering_metric = file.replace('_standardised.csv', '')
            std_df = pd.read_csv(Path(w_dir) / 'ranking' / file)
            numeric_cols = std_df.select_dtypes(include='number').columns
            std_df_grouped = std_df.groupby('ID')[numeric_cols].mean().reset_index()
            merged_df = pd.merge(std_df_grouped, original_df, on='ID')
            # Calculate EF for each scoring function and clustering metric
            for col in merged_df.columns:
                if col not in ['ID', 'Activity']:
                    sorted_df = merged_df.sort_values(col, ascending=False)
                    N10_percent = round(0.10 * len(sorted_df))
                    N1_percent = round(0.01 * len(sorted_df))
                    N100_percent = len(merged_df)
                    Hits10_percent = sorted_df.head(N10_percent)['Activity'].sum()
                    Hits1_percent = sorted_df.head(N1_percent)['Activity'].sum()
                    Hits100_percent = sorted_df['Activity'].sum()
                    ef10 = round((Hits10_percent / N10_percent) * (N100_percent / Hits100_percent), 2)
                    ef1 = round((Hits1_percent / N1_percent) * (N100_percent / Hits100_percent), 2)
                    EF_results.loc[len(EF_results)] = [col, clustering_metric, ef10, ef1]
    # Create 'consensus' directory if it doesn't exist
    (Path(w_dir) / 'consensus').mkdir(parents=True, exist_ok=True)
    # Save EF results as a CSV file
    EF_results.to_csv(Path(w_dir) / 'consensus' / 'EF_single_functions.csv',index=False)
