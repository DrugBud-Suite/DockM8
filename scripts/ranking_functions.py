import pandas as pd
import numpy as np
import functools
from scripts.utilities import create_temp_folder
import itertools
from tqdm import tqdm
from scripts.utilities import *
from IPython.display import display
from math import comb
from concurrent.futures import ProcessPoolExecutor

def standardize_scores(dataframe, clustering_metric):
    """
    Applies min-max standardisation to each column in the input dataframe using a specific rescoring function 
    determined by the column name. Returns a new dataframe containing only the standardized columns and 
    the Pose ID column, sorted alphabetically by column name.

    Args:
        dataframe: A pandas DataFrame containing numerical data to be standardized.
        clustering_metric: A string indicating the clustering metric used in the analysis.

    Returns:
        A pandas DataFrame containing only the standardized columns and the Pose ID column, sorted alphabetically 
        by column name.

    Example:
        standardize_scores(df, 'K-means')
    """
    def min_max_standardisation(score, best_value):
        if best_value == 'max':
            standardized_scores = (score - score.min()) / (score.max() - score.min())
        else:
            standardized_scores = (score.max() - score) / (score.max() - score.min())
        return standardized_scores
    rescoring_functions_standardization = {'GNINA_Affinity':'min', 
                                           'GNINA_CNN_Score':'max', 
                                           'GNINA_CNN_Affinity':'max', 
                                           'CNN_VS':'max',
                                           'Vinardo_Affinity':'min', 
                                           'AD4_Affinity':'min',
                                           'LinF9_Affinity':'min',  
                                           'RFScoreVS':'max', 
                                           'PLP':'min', 
                                           'CHEMPLP':'min', 
                                           'NNScore':'max', 
                                           'PLECnn':'max',
                                           'AAScore':'min',
                                           'ECIF':'max',
                                           'SCORCH':'max',
                                           'SCORCH_pose_score':'max'}
    for col in dataframe.columns:
        if col != 'Pose ID':
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            dataframe[f'{col}_S_{clustering_metric}'] = min_max_standardisation(dataframe[col], rescoring_functions_standardization[col])
    dataframe = dataframe.drop([col for col in dataframe.columns if col != 'Pose ID' and '_S_' not in col], axis=1)
    return dataframe[sorted(dataframe.columns)]

def rank_scores(dataframe, clustering_metric):
    dataframe = dataframe.assign(**{f'{col}_RANK': dataframe[col].rank(method='average', ascending=False) for col in dataframe.columns if col not in ['Pose ID', 'ID']})
    dataframe = dataframe.drop([col for col in dataframe.columns if col != 'Pose ID' and 'RANK' not in col], axis=1)
    return dataframe[sorted(dataframe.columns)]

def standardise_dataframes(w_dir, rescoring_folders):
    rescored_dataframes = {name: pd.read_csv(w_dir + f'/temp/{rescoring_folders[name]}/allposes_rescored.csv') for name in rescoring_folders}
    standardised_dataframes = {name + '_standardised': standardize_scores(rescored_dataframes[name], name) for name in rescoring_folders}
    return standardised_dataframes

def rank_dataframes(standardised_dataframes):
    ranked_dataframes = {name.replace('_standardised', '_ranked'): rank_scores(standardised_dataframes[name], name) for name in standardised_dataframes}
    return ranked_dataframes

def method1_ECR_best(df, clustering_metric, selected_columns):
    '''
    A method that calculates the ECR (Exponential Consensus Ranking) score for each ID in the rescored dataframe and returns the ID for the pose with the best ECR rank.
    '''
    sigma = 0.05 * len(df)
    df = df.apply(lambda x: (np.exp(-(x/sigma))/sigma)*1000 if x.name in selected_columns else x)
    df[f'Method1_ECR_{clustering_metric}'] = df.sum(axis=1, numeric_only=True)
    df = df.drop(selected_columns, axis=1)
    #Aggregate rows using best ECR per ID
    df2 = df.sort_values(f'Method1_ECR_{clustering_metric}', ascending=False).drop_duplicates(['ID'])
    return df2[['ID', f'Method1_ECR_{clustering_metric}']]

def method2_ECR_average(df, clustering_metric, selected_columns):
    '''
    A method that calculates the ECR (Exponential Consensus Ranking) score for each ID in the rescored dataframe and returns the ID along with the average ECR rank accross the clustered poses.
    '''
    sigma = 0.05 * len(df)
    df = df.apply(lambda x: (np.exp(-(x/sigma))/sigma)*1000 if x.name in selected_columns else x)
    df[f'Method2_ECR_{clustering_metric}'] = df.sum(axis=1, numeric_only=True)
    df = df.drop(selected_columns, axis=1)
    #Aggregate rows using mean ECR per ID
    df2 = df.groupby('ID', as_index=False).mean(numeric_only=True)
    return df2[['ID', f'Method2_ECR_{clustering_metric}']]

def method3_avg_ECR(df, clustering_metric, selected_columns):
    '''
    A method that first calculates the average ranks for each pose in filtered dataframe (by ID) then calculates the ECR (Exponential Consensus Ranking) for the averaged ranks.
    '''
    #Aggregate rows using mean rank per ID
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    df[selected_columns] = df[selected_columns].rank(method='average',ascending=1)
    sigma = 0.05 * len(df)
    df[selected_columns] = df[selected_columns].apply(lambda x: (np.exp(-(x/sigma))/sigma)*1000)
    df[f'Method3_ECR_{clustering_metric}'] = df.sum(axis=1, numeric_only=True)
    return df[['ID', f'Method3_ECR_{clustering_metric}']]

def method4_RbR(df, clustering_metric, selected_columns):
    '''
    A method that calculates the Rank by Rank consensus.
    '''
    df = df[['ID'] + selected_columns]
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    df[f'Method4_RbR_{clustering_metric}'] = df[selected_columns].mean(axis=1)
    
    return df[['ID', f'Method4_RbR_{clustering_metric}']]

def method5_RbV(df, clustering_metric, selected_columns):
    '''
    A method that calculates the Rank by Vote consensus.
    '''
    df['vote'] = 0
    for c in selected_columns:
        df['vote'] += (df[c] > df[c].quantile(0.95)).astype(int)
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    df[f'Method5_RbV_{clustering_metric}'] = df.mean(axis=1, numeric_only=True)
    return df[['ID', f'Method5_RbV_{clustering_metric}']]

def method6_Zscore_best(df, clustering_metric, selected_columns):
    '''
    Calculates the Z-score consensus scores for each row in the given DataFrame,
    and aggregates rows by selecting the pose with the best Z-score for each ID.
    '''
    df[selected_columns] = df[selected_columns].apply(pd.to_numeric, errors='coerce')
    z_scores = (df[selected_columns] - df[selected_columns].mean())/df[selected_columns].std()
    consensus_scores = z_scores.mean(axis=1)
    df[f'Method6_Zscore_{clustering_metric}'] = consensus_scores
    #Aggregate rows using best Z-score per ID
    df = df.sort_values(f'Method6_Zscore_{clustering_metric}', ascending=False).drop_duplicates(['ID'])
    df.set_index('ID')
    return df[['ID', f'Method6_Zscore_{clustering_metric}']]

def method7_Zscore_avg(df, clustering_metric, selected_columns):
    '''
    Calculates the Z-score consensus scores for each row in the given DataFrame,
    and aggregates rows by averaging the Z-score for each ID.
    '''
    df[selected_columns] = df[selected_columns].apply(pd.to_numeric, errors='coerce')
    z_scores = (df[selected_columns] - df[selected_columns].mean())/df[selected_columns].std()
    consensus_scores = z_scores.mean(axis=1)
    df[f'Method7_Zscore_{clustering_metric}'] = consensus_scores
    #Aggregate rows using avg Z-score per ID
    df = df.groupby('ID', as_index=False).mean(numeric_only=True)
    return df[['ID', f'Method7_Zscore_{clustering_metric}']]



def apply_consensus_methods(w_dir, clustering_metrics):
    create_temp_folder(w_dir+'/temp/ranking')
    rescoring_folders = {metric: f'rescoring_{metric}_clustered' for metric in clustering_metrics}
    standardised_dataframes = standardise_dataframes(w_dir, rescoring_folders)
    ranked_dataframes = rank_dataframes(standardised_dataframes)
    for name, df_dict in {'standardised': standardised_dataframes, 'ranked': ranked_dataframes}.items():
        for df_name, df in df_dict.items():
            df['ID'] = df['Pose ID'].str.split('_').str[0]
            df.to_csv(w_dir + f'/temp/ranking/{df_name}.csv', index=False)

    create_temp_folder(w_dir+'/temp/consensus')
    rank_methods = {'method1':method1_ECR_best, 'method2':method2_ECR_average, 'method3':method3_avg_ECR, 'method4':method4_RbR}
    score_methods = {'method5':method5_RbV, 'method6':method6_Zscore_best, 'method7':method7_Zscore_avg}
    analysed_dataframes = {f'{name}_{method}': rank_methods[method](ranked_dataframes[name+'_ranked'], name, [col for col in ranked_dataframes[name+'_ranked'] if col not in ['Pose ID', 'ID']]) for name in rescoring_folders for method in rank_methods}
    analysed_dataframes.update({f'{name}_{method}': score_methods[method](standardised_dataframes[name+'_standardised'], name, [col for col in standardised_dataframes[name+'_standardised'] if col not in ['Pose ID', 'ID']]) for name in rescoring_folders for method in score_methods})
    analysed_dataframes = {name: df.drop(columns="Pose ID", errors='ignore') for name, df in analysed_dataframes.items()}
    combined_all_methods_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['ID'], how='outer'), analysed_dataframes.values())
    combined_all_methods_df = combined_all_methods_df.reindex(columns=['ID'] + [col for col in combined_all_methods_df.columns if col != 'ID'])
    combined_all_methods_df.to_csv(w_dir+'/temp/consensus/method_results.csv', index=False)
    
def process_combination(combination, w_dir, name, standardised_df, ranked_df, column_mapping, rank_methods, score_methods):
    selected_columns = list(combination)
    ranked_selected_columns = [column_mapping[col] for col in selected_columns]
    subset_name = '_'.join(selected_columns)
    standardised_subset = standardised_df[['ID'] + selected_columns]
    ranked_subset = ranked_df[['ID'] + ranked_selected_columns]
    analysed_dataframes = {method: rank_methods[method](ranked_subset, name, ranked_selected_columns) for method in rank_methods}
    analysed_dataframes.update({method: score_methods[method](standardised_subset, name, selected_columns) for method in score_methods})
    result_dict = {}
    for method, df in analysed_dataframes.items():
        df = df.drop(columns="Pose ID", errors='ignore')
        df['method_name'] = f"{name}_{method}"
        df['selected_columns'] = subset_name
        result_dict[method] = df
    return result_dict

def process_combination_wrapper(args):
    return process_combination(*args)


import glob

def find_common_columns(file_list):
    common_columns = set()
    for i, file in enumerate(file_list):
        df = pd.read_csv(file, nrows=0)
        columns = set(df.columns)
        if i == 0:
            common_columns = columns
        else:
            common_columns = common_columns.intersection(columns)
    return list(common_columns)

def merge_csv_files(input_directory, output_file):
    all_files = glob.glob(os.path.join(input_directory, "*.csv"))

    common_columns = find_common_columns(all_files)
    combined_df = None

    for file in all_files:
        df = pd.read_csv(file)
        if combined_df is None:
            combined_df = df
        else:
            combined_df = combined_df.merge(df, on=common_columns, how='outer')

    combined_df.to_csv(output_file, index=False)

def apply_consensus_methods_combinations(w_dir, clustering_metrics):
    create_temp_folder(w_dir+'/temp/ranking')
    rescoring_folders = {metric: f'rescoring_{metric}_clustered' for metric in clustering_metrics}
    standardised_dataframes = standardise_dataframes(w_dir, rescoring_folders)
    ranked_dataframes = rank_dataframes(standardised_dataframes)
    for name, df_dict in {'standardised': standardised_dataframes, 'ranked': ranked_dataframes}.items():
        for df_name, df in df_dict.items():
            df['ID'] = df['Pose ID'].str.split('_').str[0]
            df.to_csv(w_dir + f'/temp/ranking/{df_name}.csv', index=False)
            
    create_temp_folder(w_dir+'/temp/consensus')
    rank_methods = {'method1':method1_ECR_best, 'method2':method2_ECR_average, 'method3':method3_avg_ECR, 'method4':method4_RbR}
    score_methods = {'method5':method5_RbV, 'method6':method6_Zscore_best, 'method7':method7_Zscore_avg}
    
    consensus_summary = pd.DataFrame()
    for name in rescoring_folders:
        standardised_df = standardised_dataframes[name+'_standardised']
        ranked_df = ranked_dataframes[name+'_ranked']
        calc_columns = [col for col in standardised_df.columns if col not in ['Pose ID', 'ID']]
        
        # Create a mapping between the column names in the standardised_df and ranked_df
        column_mapping = {col: f"{col}_RANK" for col in calc_columns}
        ranked_df = ranked_df.rename(columns=column_mapping)
        
        with ProcessPoolExecutor() as executor:
            printlog('Calculating consensus methods for every possible score combination...')
            for L in range(2, len(calc_columns)):
                combinations = list(itertools.combinations(calc_columns, L))
                total_combinations = sum(comb(len(calc_columns), L) for L in range(1, len(calc_columns) + 1))
                args = [(subset, w_dir, name, standardised_df, ranked_df, column_mapping, rank_methods, score_methods) for subset in combinations]

                for result_dict in tqdm(executor.map(process_combination_wrapper, args), total=total_combinations):
                    for method, df in result_dict.items():
                        file_path = w_dir + f'/temp/consensus/{method}_L{L}.csv'
                        if os.path.exists(file_path):
                            df.to_csv(file_path, mode='a', header=False, index=False)
                        else:
                            df.to_csv(file_path, index=False)
    merge_csv_files(w_dir+'/temp/consensus/', w_dir+'/temp/consensus/merged_csv.csv')