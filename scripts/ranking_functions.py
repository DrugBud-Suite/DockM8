import pandas as pd
import numpy as np
import functools
from scripts.utilities import create_temp_folder
from sklearn.preprocessing import StandardScaler
from IPython.display import display

def standardize_scores(dataframe, clustering_method):
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
                                            'AAScore':'min'}
    for col in dataframe.columns:
        if col != 'Pose ID':
            dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
            dataframe[f'{col}_S_{clustering_method}'] = min_max_standardisation(dataframe[col], rescoring_functions_standardization[col])
    dataframe = dataframe.drop([col for col in dataframe.columns if col != 'Pose ID' and '_S_' not in col], axis=1)
    return dataframe[sorted(dataframe.columns)]

def rank_scores(dataframe, clustering_method):
    dataframe = dataframe.assign(**{f'{col}_RANK': dataframe[col].rank(method='average', ascending=False) for col in dataframe.columns if col != 'Pose ID'})
    dataframe = dataframe.drop([col for col in dataframe.columns if col != 'Pose ID' and 'RANK' not in col], axis=1)
    return dataframe[sorted(dataframe.columns)]

def method1_ECR_best(df, clustering_method):
    calc = [col for col in df.columns if col not in ['Pose ID', 'ID']]
    sigma = 0.05 * len(df)
    df = df.apply(lambda x: (np.exp(-(x/sigma))/sigma)*1000 if x.name in calc else x)
    df[f'Method1_ECR_{clustering_method}'] = df.sum(axis=1, numeric_only=True)
    df = df.drop(calc, axis=1)
    #Aggregate rows using best ECR per ID
    df2 = df.sort_values(f'Method1_ECR_{clustering_method}', ascending=False).drop_duplicates(['ID'])
    return df2[['ID', f'Method1_ECR_{clustering_method}']]

def method2_ECR_average(df, clustering_method):
    calc = [col for col in df.columns if col not in ['Pose ID', 'ID']]
    sigma = 0.05 * len(df)
    df = df.apply(lambda x: (np.exp(-(x/sigma))/sigma)*1000 if x.name in calc else x)
    df[f'Method2_ECR_{clustering_method}'] = df.sum(axis=1, numeric_only=True)
    df = df.drop(calc, axis=1)
    #Aggregate rows using mean ECR per ID
    df2 = df.groupby('ID', as_index=False).mean(numeric_only=True)
    return df2[['ID', f'Method2_ECR_{clustering_method}']]

def method3_avg_ECR(df, clustering_method):
    calc = [col for col in df.columns if col not in ['Pose ID', 'ID']]
    #Aggregate rows using mean rank per ID
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    df[calc] = df[calc].rank(method='average',ascending=1)
    sigma = 0.05 * len(df)
    df[calc] = df[calc].apply(lambda x: (np.exp(-(x/sigma))/sigma)*1000)
    df[f'Method3_ECR_{clustering_method}'] = df.sum(axis=1, numeric_only=True)
    return df[['ID', f'Method3_ECR_{clustering_method}']]

def method4_RbR(df, clustering_method):
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    df[f'Method4_RbR_{clustering_method}'] = df.mean(axis=1, numeric_only=True)
    return df[['ID', f'Method4_RbR_{clustering_method}']]

def method5_RbV(df, clustering_method):
    calc = [col for col in df.columns if col not in ['Pose ID', 'ID']]
    df['vote'] = 0
    for c in calc:
        df['vote'] += (df[c] > df[c].quantile(0.95)).astype(int)
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    df[f'Method5_RbV_{clustering_method}'] = df.mean(axis=1, numeric_only=True)
    return df[['ID', f'Method5_RbV_{clustering_method}']]

def method6_Zscore_best(df, clustering_method):
    calc = [col for col in df.columns if col not in ['Pose ID', 'ID']]
    df[calc] = df[calc].apply(pd.to_numeric, errors='coerce')
    z_scores = (df[calc] - df[calc].mean())/df[calc].std()
    consensus_scores = z_scores.mean(axis=1)
    df[f'Method6_Zscore_{clustering_method}'] = consensus_scores
    #Aggregate rows using best Z-score per ID
    df = df.sort_values(f'Method6_Zscore_{clustering_method}', ascending=False).drop_duplicates(['ID'])
    df.set_index('ID')
    return df[['ID', f'Method6_Zscore_{clustering_method}']]

def method7_Zscore_avg(df, clustering_method):
    calc = [col for col in df.columns if col not in ['Pose ID', 'ID']]
    df[calc] = df[calc].apply(pd.to_numeric, errors='coerce')
    z_scores = (df[calc] - df[calc].mean())/df[calc].std()
    consensus_scores = z_scores.mean(axis=1)
    df[f'Method7_Zscore_{clustering_method}'] = consensus_scores
    #Aggregate rows using avg Z-score per ID
    df = df.groupby('ID', as_index=False).mean(numeric_only=True)
    return df[['ID', f'Method7_Zscore_{clustering_method}']]

def apply_consensus_methods(w_dir, clustering_metrics):
    create_temp_folder(w_dir+'/temp/ranking')
    rescoring_folders = {metric:f'rescoring_{metric}_clustered' for metric in clustering_metrics}
    rescored_dataframes = {name: pd.read_csv(w_dir+f'/temp/{rescoring_folders[name]}/allposes_rescored.csv') for name in rescoring_folders}
    standardised_dataframes = {name+'_standardised': standardize_scores(rescored_dataframes[name], name) for name in rescoring_folders}
    for name, standardised_df in standardised_dataframes.items():
        standardised_df['ID'] = standardised_df['Pose ID'].str.split('_').str[0]
        standardised_df.to_csv(w_dir+f'/temp/ranking/{name}.csv', index=False)
    ranked_dataframes = {name.replace('_standardised', '_ranked'): rank_scores(standardised_dataframes[name], name) for name in standardised_dataframes}
    for name, ranked_df in ranked_dataframes.items():
        ranked_df['ID'] = ranked_df['Pose ID'].str.split('_').str[0]
        ranked_df.to_csv(w_dir+f'/temp/ranking/{name}.csv', index=False)
    create_temp_folder(w_dir+'/temp/consensus')
    rank_methods = {'method1':method1_ECR_best, 'method2':method2_ECR_average, 'method3':method3_avg_ECR, 'method4':method4_RbR}
    score_methods = {'method5':method5_RbV, 'method6':method6_Zscore_best, 'method7':method7_Zscore_avg}
    analysed_dataframes = {f'{name}_{method}': rank_methods[method](ranked_dataframes[name+'_ranked'], name) for name in rescoring_folders for method in rank_methods}
    analysed_dataframes.update({f'{name}_{method}': score_methods[method](standardised_dataframes[name+'_standardised'], name) for name in rescoring_folders for method in score_methods})
    analysed_dataframes = {name: df.drop(columns="Pose ID", errors='ignore') for name, df in analysed_dataframes.items()}
    combined_all_methods_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['ID'], how='outer'), analysed_dataframes.values())
    combined_all_methods_df = combined_all_methods_df.reindex(columns=['ID'] + [col for col in combined_all_methods_df.columns if col != 'ID'])
    combined_all_methods_df.to_csv(w_dir+'/temp/consensus/method_results.csv', index=False)
