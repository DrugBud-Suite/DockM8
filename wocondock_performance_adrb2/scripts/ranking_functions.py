import pandas as pd
import numpy as np
import functools
from scripts.utilities import create_temp_folder

def rank(dataframe, clustering_method):
    dataframe['GNINA_Affinity_RANK_{}'.format(clustering_method)] = dataframe['GNINA_Affinity'].rank(method='average',ascending=1)
    dataframe['GNINA_CNNscore_RANK_{}'.format(clustering_method)] = dataframe['GNINA_CNN_Score'].rank(method='average',ascending=0)
    dataframe['GNINA_CNN_Affinity_RANK_{}'.format(clustering_method)] = dataframe['GNINA_CNN_Affinity'].rank(method='average',ascending=0)
    dataframe['Vinardo_Affinity_RANK_{}'.format(clustering_method)] = dataframe['Vinardo_Affinity'].rank(method='average',ascending=1)
    dataframe['AD4_Affinity_RANK_{}'.format(clustering_method)] = dataframe['AD4_Affinity'].rank(method='average',ascending=1)
    dataframe['RFScoreV1_RANK_{}'.format(clustering_method)] = dataframe['RFScoreV1'].rank(method='average',ascending=0)
    dataframe['RFScoreV2_RANK_{}'.format(clustering_method)] = dataframe['RFScoreV2'].rank(method='average',ascending=0)
    dataframe['RFScoreV3_RANK_{}'.format(clustering_method)] = dataframe['RFScoreV3'].rank(method='average',ascending=0)
    dataframe['PLP_RANK_{}'.format(clustering_method)] = dataframe['PLP'].rank(method='average',ascending=1)
    dataframe['CHEMPLP_RANK_{}'.format(clustering_method)] = dataframe['CHEMPLP'].rank(method='average',ascending=1)
    dataframe['NNScore_RANK_{}'.format(clustering_method)] = dataframe['NNScore'].rank(method='average',ascending=0)
    #dataframe['PLECnn_RANK_{}'.format(clustering_method)] = dataframe['PLECnn'].rank(method='average',ascending=0)
    output_dataframe = dataframe.copy()
    for c in output_dataframe.columns.tolist():
        if c in ['GNINA_Affinity', 'GNINA_CNN_Score', 'GNINA_CNN_Affinity', 'Vinardo_Affinity', 'AD4_Affinity', 'RFScoreV1', 'RFScoreV2', 'RFScoreV3', 'PLP', 'CHEMPLP', 'NNScore', 'PLECnn']:
            output_dataframe.drop(c, axis=1, inplace=True)
        else:
            pass
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
    df2 = df.groupby('ID', as_index=True).mean(numeric_only=True)
    return df2.sort_values('Method2_ECR_{}'.format(clustering_method), ascending=0)

def method3_avg_ECR(dataframe, clustering_method):
    df = dataframe.copy()
    #Return compound ID from Pose ID
    df[['ID', 'Software', 'Pose Number']] = df['Pose ID'].str.split("_", expand=True)
    #Drop extra columns
    df = df.drop(['Software', 'Pose Number'], axis=1)
    df = df.groupby('ID', as_index=True).mean(numeric_only=True)
    df = df.rank(method='average',ascending=1)
    #Select columns for calculations
    calc = df.columns
    #Calculate ECR
    sigma = 5
    df = df.apply(lambda x: np.exp(-(x/sigma))/sigma if x.name in df.columns else x)
    df['Method3_ECR_{}'.format(clustering_method)] = df.sum(axis=1, numeric_only=True)
    df = df.drop(calc, axis=1)
    return df.sort_values('Method3_ECR_{}'.format(clustering_method), ascending=0)

from scipy.stats import zscore
##Method 6: Calculate Zscore for each column, then average across columns, then best avg Zscore
def method6_Zscore_best(dataframe, clustering_method):
    df = dataframe.copy()
    #Select columns for calculations and convert to float
    dfcolumns = df.columns
    calc = dfcolumns[1:]
    for i in calc.tolist():
        if i == 'Pose ID':
            pass
        if df[i].dtypes is not float:
            df[i] = pd.to_numeric(df[i])
        else:
            pass
    df = df.apply(lambda x: zscore(x) if x.name in calc else x)
    df['Method6_Avg_Z-score_{}'.format(clustering_method)]= df.mean(axis=1, numeric_only=True)
    #Return compound ID from Pose ID
    df[['ID', 'Software', 'Pose Number']] = df['Pose ID'].str.split("_", expand=True)
    #Drop extra columns
    df = df.drop(['Software', 'Pose Number'], axis=1)
    df = df.drop(calc, axis=1)
    #Aggregate rows using best Avg_Z-score per pose
    df2 = df.sort_values('Method6_Avg_Z-score_{}'.format(clustering_method), ascending=False).drop_duplicates(['ID'])
    df2.set_index('ID')
    return df2

def method7_Zscore_avg(dataframe, clustering_method):
    df = dataframe.copy()
    #Select columns for calculations and convert to float
    dfcolumns = df.columns
    calc = dfcolumns[1:]
    for i in calc.tolist():
        if i == 'Pose ID':
            pass
        if df[i].dtypes is not float:
            df[i] = pd.to_numeric(df[i])
        else:
            pass
    df = df.apply(lambda x: zscore(x) if x.name in calc else x)
    df['Method7_Avg_Z-score_{}'.format(clustering_method)]= df.mean(axis=1, numeric_only=True)
    #Return compound ID from Pose ID
    df[['ID', 'Software', 'Pose Number']] = df['Pose ID'].str.split("_", expand=True)
    #Drop extra columns
    df = df.drop(['Software', 'Pose Number'], axis=1)
    df = df.drop(calc, axis=1)
    #Aggregate rows using avg Z-score per pose
    df2 = df.groupby('ID', as_index=True).mean(numeric_only=True)
    return df2

def apply_all_score_methods(dataframe, clustering_method):
    method6 = method6_Zscore_best(dataframe, clustering_method)
    method7 = method7_Zscore_avg(dataframe, clustering_method)
    return method6, method7

def apply_all_rank_methods(dataframe, clustering_method):
    method1 = method1_ECR_best(dataframe, clustering_method)
    method2 = method2_ECR_average(dataframe, clustering_method)
    method3 = method3_avg_ECR(dataframe, clustering_method)
    return method1, method2, method3

def apply_ranking_methods(w_dir):
    RMSD_rescored = pd.read_csv(w_dir+'/temp/rescoring_RMSD_kS_full/allposes_rescored.csv')
    espsim_rescored = pd.read_csv(w_dir+'/temp/rescoring_espsim_d_kS_full/allposes_rescored.csv')
    spyRMSD_rescored = pd.read_csv(w_dir+'/temp/rescoring_spyRMSD_kS_full/allposes_rescored.csv')
    USRCAT_rescored = pd.read_csv(w_dir+'/temp/rescoring_usr_kS_full/allposes_rescored.csv')
    RMSD_rescored.drop(columns=RMSD_rescored.columns[0], axis=1, inplace=True)
    espsim_rescored.drop(columns=espsim_rescored.columns[0], axis=1, inplace=True)
    spyRMSD_rescored.drop(columns=spyRMSD_rescored.columns[0], axis=1, inplace=True)
    USRCAT_rescored.drop(columns=USRCAT_rescored.columns[0], axis=1, inplace=True)
    
    RMSD_rescored_ranked = rank(RMSD_rescored, 'RMSD')
    espsim_rescored_ranked = rank(espsim_rescored, 'espsim')
    spyRMSD_rescored_ranked = rank(spyRMSD_rescored, 'spyRMSD')
    USRCAT_rescored_ranked = rank(USRCAT_rescored, 'USRCAT')

    create_temp_folder(w_dir+'/temp/ranking')

    ranked_dataframes = [RMSD_rescored_ranked, espsim_rescored_ranked,spyRMSD_rescored_ranked, USRCAT_rescored_ranked]
    rescored_dataframes = [RMSD_rescored, espsim_rescored,spyRMSD_rescored, USRCAT_rescored]

    final_dataframe_ranked = pd.DataFrame()
    method1_RMSD, method2_RMSD, method3_RMSD = apply_all_rank_methods(RMSD_rescored_ranked, 'RMSD')
    method1_espsim, method2_espsim, method3_espsim = apply_all_rank_methods(espsim_rescored_ranked, 'espsim')
    method1_spyRMSD, method2_spyRMSD, method3_spyRMSD = apply_all_rank_methods(spyRMSD_rescored_ranked, 'spyRMSD')
    method1_USRCAT, method2_USRCAT, method3_USRCAT = apply_all_rank_methods(USRCAT_rescored_ranked, 'USRCAT')
    methods_rank_dfs = [method1_RMSD, method2_RMSD, method3_RMSD, method1_espsim, method2_espsim, method3_espsim, method1_spyRMSD, method2_spyRMSD, method3_spyRMSD, method1_USRCAT, method2_USRCAT, method3_USRCAT]
    for x in methods_rank_dfs:
        try:
            x.drop('Pose ID', axis=1, inplace=True)
        except:
            pass
    combined_rank_methods_df = functools.reduce(lambda  left,right: pd.merge(left,right,on=['ID'],how='outer'), methods_rank_dfs)

    method6_RMSD, method7_RMSD = apply_all_score_methods(RMSD_rescored_ranked, 'RMSD')
    method6_espsim, method7_espsim = apply_all_score_methods(espsim_rescored_ranked, 'espsim')
    method6_spyRMSD, method7_spyRMSD = apply_all_score_methods(spyRMSD_rescored_ranked, 'spyRMSD')
    method6_USRCAT, method7_USRCAT = apply_all_score_methods(USRCAT_rescored_ranked, 'USRCAT')
    methods_score_dfs = [method6_RMSD, method7_RMSD, method6_espsim, method7_espsim, method6_spyRMSD, method7_spyRMSD, method6_USRCAT, method7_USRCAT]
    for x in methods_score_dfs:
        try:
            x.drop('Pose ID', axis=1, inplace=True)
        except:
            pass
    combined_score_methods_df = functools.reduce(lambda  left,right: pd.merge(left,right,on=['ID'],how='outer'), methods_score_dfs)

    combined_all_methods_df = functools.reduce(lambda  left,right: pd.merge(left,right,on=['ID'],how='outer'), [combined_rank_methods_df, combined_score_methods_df])
    reindexed_combined_all_methods_df = combined_all_methods_df.reindex(columns=['ID', 'Method1_ECR_RMSD', 'Method2_ECR_RMSD', 'Method3_ECR_RMSD','Method1_ECR_espsim', 'Method2_ECR_espsim', 'Method3_ECR_espsim','Method1_ECR_spyRMSD', 'Method2_ECR_spyRMSD', 'Method3_ECR_spyRMSD','Method1_ECR_USRCAT', 'Method2_ECR_USRCAT', 'Method3_ECR_USRCAT','Method6_Avg_Z-score_RMSD', 'Method7_Avg_Z-score_RMSD','Method6_Avg_Z-score_espsim', 'Method7_Avg_Z-score_espsim','Method6_Avg_Z-score_spyRMSD', 'Method7_Avg_Z-score_spyRMSD','Method6_Avg_Z-score_USRCAT', 'Method7_Avg_Z-score_USRCAT'])
    reindexed_combined_all_methods_df.to_csv(w_dir+'/temp/ranking/ranking_results.csv', index=False)