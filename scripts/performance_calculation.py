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
    rescored_dataframes = {name: pd.read_csv(Path(w_dir) / rescoring_folders[name] / 'allposes_rescored.csv') for name in rescoring_folders}
    standardised_dataframes = {f'{name}_standardised': standardize_scores(rescored_dataframes[name]) for name in rescoring_folders}
    ranked_dataframes = {f'{name}_ranked': rank_scores(standardised_dataframes[f'{name}_standardised']) for name in rescoring_folders}
    return standardised_dataframes, ranked_dataframes

def process_combination(combination, w_dir, name, standardised_df, ranked_df, column_mapping, rank_methods, score_methods, docking_library, original_df):
    selected_columns = list(combination)
    ranked_selected_columns = [column_mapping[col] for col in selected_columns]
    subset_name = '_'.join(selected_columns)
    replacements_dict = {'_R_': '', '_S_': '_'}
    for key, value in replacements_dict.items():
        subset_name = subset_name.replace(key, value)
        
    standardised_subset = standardised_df[['ID'] + selected_columns]
    ranked_subset = ranked_df[['ID'] + ranked_selected_columns]
    
    analysed_dataframes = {method: rank_methods[method](ranked_subset, name, ranked_selected_columns) for method in rank_methods}
    analysed_dataframes.update({method: score_methods[method](standardised_subset, name, selected_columns) for method in score_methods})

    def calculate_EF1(df, w_dir, docking_library, original_df):
        # Calculate EFs for consensus methods
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
        # Create a new dataframe with the method name, selected columns, and
        # enrichment factor
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
    (Path(w_dir) / 'ranking').mkdir(parents=True, exist_ok=True)
    rescoring_folders = {metric: f'rescoring_{metric}_clustered' for metric in clustering_metrics}
    standardised_dataframes, ranked_dataframes = process_dataframes(w_dir, rescoring_folders)

    for name, df_dict in {'standardised': standardised_dataframes, 'ranked': ranked_dataframes}.items():
        for df_name, df in df_dict.items():
            df['ID'] = df['Pose ID'].str.split('_').str[0]
            df.to_csv(Path(w_dir) / 'ranking' / f'{df_name}.csv', index=False)

    (Path(w_dir) / 'consensus').mkdir(parents=True, exist_ok=True)
    rank_methods = {
        'method1': method1_ECR_best,
        'method2': method2_ECR_average,
        'method3': method3_avg_ECR,
        'method4': method4_RbR}
    score_methods = {
        'method5': method5_RbV,
        'method6': method6_Zscore_best,
        'method7': method7_Zscore_avg}

    print('Loading library')
    original_df = PandasTools.LoadSDF(str(docking_library), molColName=None, idName='ID')
    original_df = original_df[['ID', 'Activity']]
    original_df['Activity'] = pd.to_numeric(original_df['Activity'])
    df_list = []

    printlog('Calculating consensus methods for every possible score combination...')

    for name in tqdm(rescoring_folders, total=len(rescoring_folders)):
        standardised_df = standardised_dataframes[name + '_standardised']
        ranked_df = ranked_dataframes[name + '_ranked']
        calc_columns = [col for col in standardised_df.columns if col not in ['Pose ID', 'ID']]
        column_mapping = {col: f"{col}_R" for col in calc_columns}
        ranked_df = ranked_df.rename(columns=column_mapping)
        parallel = Parallel(n_jobs=int(os.cpu_count() - 2), backend='multiprocessing')

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
    (Path(w_dir) / 'ranking').mkdir(parents=True, exist_ok=True)
    rescoring_folders = {metric: f'rescoring_{metric}_clustered' for metric in clustering_metrics}
    
    standardised_dataframes, ranked_dataframes = process_dataframes(w_dir, rescoring_folders)

    for name, df_dict in {'standardised': standardised_dataframes, 'ranked': ranked_dataframes}.items():
        for df_name, df in df_dict.items():
            df['ID'] = df['Pose ID'].str.split('_').str[0]
            df.to_csv(Path(w_dir) / 'ranking' / f'{df_name}.csv',index=False)

    original_df = PandasTools.LoadSDF(str(docking_library), molColName=None, idName='ID')
    original_df = original_df[['ID', 'Activity']]
    original_df['Activity'] = pd.to_numeric(original_df['Activity'])
    EF_results = pd.DataFrame(columns=['Scoring Function', 'Clustering Metric', 'EF10%', 'EF1%'])

    # Calculate EFs for separate scoring functions
    for file in os.listdir(Path(w_dir) / 'ranking'):
        if file.endswith('_standardised.csv'):
            clustering_metric = file.replace('_standardised.csv', '')
            std_df = pd.read_csv(Path(w_dir) / 'ranking' / file)
            numeric_cols = std_df.select_dtypes(include='number').columns
            std_df_grouped = std_df.groupby('ID')[numeric_cols].mean().reset_index()
            merged_df = pd.merge(std_df_grouped, original_df, on='ID')

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

    (Path(w_dir) / 'consensus').mkdir(parents=True, exist_ok=True)
    EF_results.to_csv(Path(w_dir) / 'consensus' / 'EF_single_functions.csv',index=False)
