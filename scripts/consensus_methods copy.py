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
from joblib import Parallel, delayed


def method1_ECR_best(df, clustering_metric, selected_columns):
    '''
    A method that calculates the ECR (Exponential Consensus Ranking) score for each ID in the rescored dataframe and returns the ID for the pose with the best ECR rank.
    '''
    sigma = 0.05 * len(df)
    df = df.apply(lambda x: (np.exp(-(x / sigma)) / sigma) *
                  1000 if x.name in selected_columns else x)
    df[f'Method1_ECR_{clustering_metric}'] = df.sum(axis=1, numeric_only=True)
    df = df.drop(selected_columns, axis=1)
    # Aggregate rows using best ECR per ID
    df2 = df.sort_values(
        f'Method1_ECR_{clustering_metric}',
        ascending=False).drop_duplicates(
        ['ID'])
    # Standardize the final output column
    max_score = df2[f'Method1_ECR_{clustering_metric}'].max()
    min_score = df2[f'Method1_ECR_{clustering_metric}'].min()
    df2[f'Method1_ECR_{clustering_metric}'] = (
        df2[f'Method1_ECR_{clustering_metric}'] - min_score) / (max_score - min_score)
    return df2[['ID', f'Method1_ECR_{clustering_metric}']]


def method2_ECR_average(df, clustering_metric, selected_columns):
    '''
    A method that calculates the ECR (Exponential Consensus Ranking) score for each ID in the rescored dataframe and returns the ID along with the average ECR rank accross the clustered poses.
    '''
    sigma = 0.05 * len(df)
    df = df.apply(lambda x: (np.exp(-(x / sigma)) / sigma) *
                  1000 if x.name in selected_columns else x)
    df[f'Method2_ECR_{clustering_metric}'] = df.sum(axis=1, numeric_only=True)
    df = df.drop(selected_columns, axis=1)
    # Aggregate rows using mean ECR per ID
    df2 = df.groupby('ID', as_index=False).mean(numeric_only=True)
    # Standardize the final output column
    max_score = df2[f'Method2_ECR_{clustering_metric}'].max()
    min_score = df2[f'Method2_ECR_{clustering_metric}'].min()
    df2[f'Method2_ECR_{clustering_metric}'] = (
        df2[f'Method2_ECR_{clustering_metric}'] - min_score) / (max_score - min_score)

    return df2[['ID', f'Method2_ECR_{clustering_metric}']]


def method3_avg_ECR(df, clustering_metric, selected_columns):
    '''
    A method that first calculates the average ranks for each pose in filtered dataframe (by ID) then calculates the ECR (Exponential Consensus Ranking) for the averaged ranks.
    '''
    # Aggregate rows using mean rank per ID
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    df[selected_columns] = df[selected_columns].rank(
        method='average', ascending=1)
    sigma = 0.05 * len(df)
    df[selected_columns] = df[selected_columns].apply(
        lambda x: (np.exp(-(x / sigma)) / sigma) * 1000)
    df[f'Method3_ECR_{clustering_metric}'] = df.sum(axis=1, numeric_only=True)
    # Standardize the final output column
    max_score = df[f'Method3_ECR_{clustering_metric}'].max()
    min_score = df[f'Method3_ECR_{clustering_metric}'].min()
    df[f'Method3_ECR_{clustering_metric}'] = (
        df[f'Method3_ECR_{clustering_metric}'] - min_score) / (max_score - min_score)

    return df[['ID', f'Method3_ECR_{clustering_metric}']]


def method4_RbR(df, clustering_metric, selected_columns):
    '''
    A method that calculates the Rank by Rank consensus.
    '''
    df = df[['ID'] + selected_columns]
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    df[f'Method4_RbR_{clustering_metric}'] = df[selected_columns].mean(axis=1)

    # Standardize the final column
    min_value = df[f'Method4_RbR_{clustering_metric}'].min()
    max_value = df[f'Method4_RbR_{clustering_metric}'].max()
    df[f'Method4_RbR_{clustering_metric}'] = (
        max_value - df[f'Method4_RbR_{clustering_metric}']) / (max_value - min_value)

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

    # Standardize the final column from 0 to 1
    max_value = df[f'Method5_RbV_{clustering_metric}'].max()
    min_value = df[f'Method5_RbV_{clustering_metric}'].min()
    df[f'Method5_RbV_{clustering_metric}'] = (
        df[f'Method5_RbV_{clustering_metric}'] - min_value) / (max_value - min_value)

    return df[['ID', f'Method5_RbV_{clustering_metric}']]


def method6_Zscore_best(df, clustering_metric, selected_columns):
    '''
    Calculates the Z-score consensus scores for each row in the given DataFrame,
    and aggregates rows by selecting the pose with the best Z-score for each ID.
    '''
    df[selected_columns] = df[selected_columns].apply(
        pd.to_numeric, errors='coerce')
    z_scores = (df[selected_columns] - df[selected_columns].mean()
                ) / df[selected_columns].std()
    consensus_scores = z_scores.mean(axis=1)
    df[f'Method6_Zscore_{clustering_metric}'] = consensus_scores
    # Aggregate rows using best Z-score per ID
    df = df.sort_values(
        f'Method6_Zscore_{clustering_metric}',
        ascending=False).drop_duplicates(
        ['ID'])
    df.set_index('ID')
    max_value = df[f'Method6_Zscore_{clustering_metric}'].max()
    min_value = df[f'Method6_Zscore_{clustering_metric}'].min()
    df[f'Method6_Zscore_{clustering_metric}'] = (
        df[f'Method6_Zscore_{clustering_metric}'] - min_value) / (max_value - min_value)
    return df[['ID', f'Method6_Zscore_{clustering_metric}']]


def method7_Zscore_avg(df, clustering_metric, selected_columns):
    '''
    Calculates the Z-score consensus scores for each row in the given DataFrame,
    and aggregates rows by averaging the Z-score for each ID.
    '''
    df[selected_columns] = df[selected_columns].apply(
        pd.to_numeric, errors='coerce')
    z_scores = (df[selected_columns] - df[selected_columns].mean()
                ) / df[selected_columns].std()
    consensus_scores = z_scores.mean(axis=1)
    df[f'Method7_Zscore_{clustering_metric}'] = consensus_scores
    # Aggregate rows using avg Z-score per ID
    df = df.groupby('ID', as_index=False).mean(numeric_only=True)
    max_value = df[f'Method7_Zscore_{clustering_metric}'].max()
    min_value = df[f'Method7_Zscore_{clustering_metric}'].min()
    df[f'Method7_Zscore_{clustering_metric}'] = (
        df[f'Method7_Zscore_{clustering_metric}'] - min_value) / (max_value - min_value)
    return df[['ID', f'Method7_Zscore_{clustering_metric}']]
