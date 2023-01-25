import pandas as pd
import numpy as np
import functools
from scripts.utilities import create_temp_folder
from scipy.stats import zscore
from IPython.display import display
from sklearn.preprocessing import StandardScaler

def rank_simplified(dataframe, clustering_method):
    columns_to_rank = ['GNINA_Affinity', 'GNINA_CNN_Score', 'GNINA_CNN_Affinity', 'Vinardo_Affinity', 'AD4_Affinity', 'RFScoreV1', 'RFScoreV2', 'RFScoreV3', 'PLP', 'CHEMPLP', 'NNScore', 'PLECnn']
    for col in columns_to_rank:
        dataframe['{}_RANK_{}'.format(col, clustering_method)] = dataframe[col].rank(method='average', ascending=(col not in ['GNINA_Affinity', 'Vinardo_Affinity', 'AD4_Affinity', 'PLP', 'CHEMPLP']))
    output_dataframe = dataframe.drop(columns_to_rank, axis=1)
    return output_dataframe

def method1_ECR_best(dataframe, clustering_method):
    df = dataframe.copy()
    #Select columns for calculations
    dfcolumns = df.columns
    calc = dfcolumns[1:]
    #Calculate ECR
    sigma = 5
    df = df.apply(lambda x: np.exp(-(x/sigma))/sigma if x.name in calc else x)
    df['Method1_ECR_{}'.format(clustering_method)] = df.sum(axis=1, numeric_only=True)
    #Return compound ID from Pose ID
    df[['ID', 'Software', 'Pose Number']] = df['Pose ID'].str.split("_", expand=True)
    #Drop extra columns
    df = df.drop(['Software', 'Pose Number'], axis=1)
    df = df.drop(calc, axis=1)
    #Aggregate rows using best ECR per pose
    df2 = df.sort_values('Method1_ECR_{}'.format(clustering_method), ascending=False).drop_duplicates(['ID'])
    df2.set_index('ID')
    return df2

def method1_ECR_best_simplified(dataframe, clustering_method):
    df = dataframe.copy()
    #Select columns for calculations
    calc = df.columns[1:]
    #Calculate ECR
    sigma = 5
    df[calc] = df[calc].apply(lambda x: np.exp(-(x/sigma))/sigma)
    df['Method1_ECR_{}'.format(clustering_method)] = df[calc].sum(axis=1)
    #Return compound ID from Pose ID
    df[['ID', 'Software', 'Pose Number']] = df['Pose ID'].str.split("_", expand=True)
    #Drop extra columns
    df.drop(['Software', 'Pose Number'], axis=1, inplace=True)
    df.drop(calc, axis=1, inplace=True)
    #Aggregate rows using best ECR per pose
    df.sort_values('Method1_ECR_{}'.format(clustering_method), ascending=False, inplace=True)
    df.drop_duplicates(['ID'], keep='first', inplace=True)
    df.set_index('ID', inplace=True)
    return df

def method2_ECR_average(dataframe, clustering_method):
    df = dataframe.copy()
    #Select columns for calculations
    dfcolumns = df.columns
    calc = dfcolumns[1:]
    #Calculate ECR
    sigma = 5
    df = df.apply(lambda x: np.exp(-(x/sigma))/sigma if x.name in calc else x)
    df['Method2_ECR_{}'.format(clustering_method)] = df.sum(axis=1, numeric_only=True)
    #Return compound ID from Pose ID
    df[['ID', 'Software', 'Pose Number']] = df['Pose ID'].str.split("_", expand=True)
    #Drop extra columns
    df = df.drop(['Software', 'Pose Number'], axis=1)
    df = df.drop(calc, axis=1)
    df2 = df.groupby('ID', as_index=False).mean(numeric_only=True)
    return df2.sort_values('Method2_ECR_{}'.format(clustering_method), ascending=0)

def method3_avg_ECR(dataframe, clustering_method):
    df = dataframe.copy()
    calc = [col for col in df.columns if col != 'Pose ID']
    #Return compound ID from Pose ID
    df[['ID', 'Software', 'Pose Number']] = df['Pose ID'].str.split("_", expand=True)
    #Drop extra columns
    df = df.drop(['Software', 'Pose Number'], axis=1)
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    df[calc] = df[calc].rank(method='average',ascending=1)
    #Calculate ECR
    sigma = 5
    df[calc] = df[calc].apply(lambda x: np.exp(-(x/sigma))/sigma)
    df['Method3_ECR_{}'.format(clustering_method)] = df.sum(axis=1, numeric_only=True)
    df = df[['ID', f'Method3_ECR_{clustering_method}']]
    return df.sort_values(f'Method3_ECR_{clustering_method}', ascending=0)

def method4_RbR(dataframe, clustering_method):
    df = dataframe.copy()
    #Return compound ID from Pose ID
    df[['ID', 'Software', 'Pose Number']] = df['Pose ID'].str.split("_", expand=True)
    #Drop extra columns
    df = df.drop(['Software', 'Pose Number'], axis=1)
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    df[f'Method4_RbR_{clustering_method}'] = df.mean(axis=1, numeric_only=True)
    df = df[['ID', f'Method4_RbR_{clustering_method}']]
    return df.sort_values([f'Method4_RbR_{clustering_method}'], ascending=0)#

def method5_RbV(dataframe, clustering_method):
    df = dataframe.copy()
    to_flip = ['GNINA_Affinity', 'Vinardo_Affinity', 'AD4_Affinity', 'PLP', 'CHEMPLP']
    df[to_flip] = -df[to_flip]
    calc = [col for col in df.columns if col != 'Pose ID']
    df['vote'] = 0
    for c in calc:
        df['vote'] += (df[c] > df[c].quantile(0.95)).astype(int)
    #Return compound ID from Pose ID
    df[['ID', 'Software', 'Pose Number']] = df['Pose ID'].str.split("_", expand=True)
    #Drop extra columns
    df = df.drop(['Software', 'Pose Number'], axis=1)
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    df[f'Method5_RbV_{clustering_method}'] = df.mean(axis=1, numeric_only=True)
    df = df[['ID', f'Method5_RbV_{clustering_method}']]
    return df.sort_values([f'Method5_RbV_{clustering_method}'], ascending=0)

def method6_Zscore_best(dataframe, clustering_method):
    df = dataframe.copy()
    scaler = StandardScaler()
    #Select columns for calculations and convert to float
    calc = [col for col in df.columns if col != 'Pose ID']
    df[calc] = df[calc].apply(pd.to_numeric, errors='coerce')
    dataframe[calc] = scaler.fit_transform(df[calc])
    z_scores = (df[calc] - df[calc].mean())/df[calc].std()
    consensus_scores = z_scores.mean(axis=1)
    df[f'method6_Zscore_{clustering_method}'] = consensus_scores
    df = df[['Pose ID', f'method6_Zscore_{clustering_method}']]
    #Return compound ID from Pose ID
    df[['ID', 'Software', 'Pose Number']] = df['Pose ID'].str.split("_", expand=True)
    #Drop extra columns
    df = df.drop(['Software', 'Pose Number'], axis=1)
    #Aggregate rows using best Avg_Z-score per pose
    df = df.sort_values(f'method6_Zscore_{clustering_method}', ascending=False).drop_duplicates(['ID'])
    df.set_index('ID')
    return df

def method7_Zscore_avg(dataframe, clustering_method):
    df = dataframe.copy()
    scaler = StandardScaler()
    #Select columns for calculations and convert to float
    calc = [col for col in df.columns if col != 'Pose ID']
    df[calc] = df[calc].apply(pd.to_numeric, errors='coerce')
    dataframe[calc] = scaler.fit_transform(df[calc])
    z_scores = (df[calc] - df[calc].mean())/df[calc].std()
    consensus_scores = z_scores.mean(axis=1)
    df[f'method7_Zscore_{clustering_method}'] = consensus_scores
    df = df[['Pose ID', f'method7_Zscore_{clustering_method}']]
    #Return compound ID from Pose ID
    df[['ID', 'Software', 'Pose Number']] = df['Pose ID'].str.split("_", expand=True)
    #Drop extra columns
    df = df.drop(['Software', 'Pose Number'], axis=1)
    #Aggregate rows using best Avg_Z-score per pose
    df = df.groupby('ID', as_index=False).mean(numeric_only=True)
    return df

def method8_Zscore_flipped_best(dataframe, clustering_method):
    df = dataframe.copy()
    scaler = StandardScaler()
    #Select columns for calculations and convert to float
    calc = [col for col in df.columns if col != 'Pose ID']
    df[calc] = df[calc].apply(pd.to_numeric, errors='coerce')
    to_flip = ['GNINA_Affinity', 'Vinardo_Affinity', 'AD4_Affinity', 'PLP', 'CHEMPLP']
    df[to_flip] = df[to_flip]*-1
    dataframe[calc] = scaler.fit_transform(df[calc])
    z_scores = (df[calc] - df[calc].mean())/df[calc].std()
    consensus_scores = z_scores.mean(axis=1)
    df[f'method8_Zscore_{clustering_method}'] = consensus_scores
    df = df[['Pose ID', f'method8_Zscore_{clustering_method}', ]]
    #Return compound ID from Pose ID
    df[['ID', 'Software', 'Pose Number']] = df['Pose ID'].str.split("_", expand=True)
    #Drop extra columns
    df = df.drop(['Software', 'Pose Number'], axis=1)
    #Aggregate rows using best Avg_Z-score per pose
    df = df.sort_values(f'method8_Zscore_{clustering_method}', ascending=False).drop_duplicates(['ID'])
    df.set_index('ID')
    return df

def method9_Zscore_flipped_avg(dataframe, clustering_method):
    df = dataframe.copy()
    scaler = StandardScaler()
    #Select columns for calculations and convert to float
    calc = [col for col in df.columns if col != 'Pose ID']
    df[calc] = df[calc].apply(pd.to_numeric, errors='coerce')
    to_flip = ['GNINA_Affinity', 'Vinardo_Affinity', 'AD4_Affinity', 'PLP', 'CHEMPLP']
    df[to_flip] = df[to_flip]*-1
    df[calc] = scaler.fit_transform(df[calc])
    z_scores = (df[calc] - df[calc].mean())/df[calc].std()
    consensus_scores = z_scores.mean(axis=1)
    df[f'method9_Zscore_{clustering_method}'] = consensus_scores
    df = df[['Pose ID', f'method9_Zscore_{clustering_method}', ]]
    #Return compound ID from Pose ID
    df[['ID', 'Software', 'Pose Number']] = df['Pose ID'].str.split("_", expand=True)
    #Drop extra columns
    df = df.drop(['Software', 'Pose Number'], axis=1)
    #Aggregate rows using best Avg_Z-score per pose
    df = df.groupby('ID', as_index=False).mean(numeric_only=True)
    return df

def apply_ranking_methods_simplified(w_dir):
    create_temp_folder(w_dir+'/temp/ranking')
    ranking_filenames = {'RMSD': 'rescoring_RMSD_clustered', 'espsim': 'rescoring_espsim_clustered', 'spyRMSD': 'rescoring_spyRMSD_clustered', 'USRCAT': 'rescoring_USRCAT_clustered', '3DScore': 'rescoring_3DScore_clustered', 'bestpose': 'rescoring_bestpose_clustered'}
    rescored_dataframes = {name: pd.read_csv(w_dir+f'/temp/{ranking_filenames[name]}/allposes_rescored.csv') for name in ranking_filenames}
    for df in rescored_dataframes.values():
        df.drop(columns=df.columns[0], axis=1, inplace=True)
    ranked_dataframes = {name+'_ranked': rank_simplified(rescored_dataframes[name], name) for name in ranking_filenames}
    rank_methods = {'method1':method1_ECR_best, 'method2':method2_ECR_average, 'method3':method3_avg_ECR, 'method4':method4_RbR}
    score_methods = {'method5':method5_RbV, 'method6':method6_Zscore_best, 'method7':method7_Zscore_avg, 'method8':method8_Zscore_flipped_best, 'method9':method9_Zscore_flipped_avg}
    
    analysed_dataframes = {f'{name}_{method}': rank_methods[method](ranked_dataframes[name+'_ranked'], name) for name in ranking_filenames for method in rank_methods}
    analysed_dataframes.update({f'{name}_{method}': score_methods[method](rescored_dataframes[name], name) for name in ranking_filenames for method in score_methods})
    analysed_dataframes = {name: df.drop(columns="Pose ID", errors='ignore') for name, df in analysed_dataframes.items()}
    combined_all_methods_df = functools.reduce(lambda left, right: pd.merge(left, right, on=['ID'], how='outer'), analysed_dataframes.values())
    combined_all_methods_df = combined_all_methods_df.reindex(columns=['ID'] + [col for col in combined_all_methods_df.columns if col != 'ID'])
    combined_all_methods_df.to_csv(w_dir+'/temp/ranking/ranking_results.csv', index=False)