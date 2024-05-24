import warnings

import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def RbR_best(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
    """
    Calculates the Rank by Rank (RbR) consensus score for each ID in the input dataframe.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the input data with columns 'ID', 'Score1', 'Score2', and so on.
        clustering_metric (str): A string representing the clustering metric used.
        selected_columns (list): A list of strings representing the selected columns for calculating the RbR score.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns 'ID' and 'RbR_{clustering_metric}', where 'RbR_{clustering_metric}' represents the RbR score for each ID.
    """
    # Select the 'ID' column and the selected columns from the input dataframe
    df = df[['ID'] + selected_columns]
    # Convert selected columns to numeric values
    df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors='coerce')
    # Calculate the mean rank across the selected columns
    df['RbR'] = df[selected_columns].mean(axis=1)
    df = df[['ID', 'RbR']]
    # Sort the dataframe by the mean rank in ascending order
    df = df.sort_values('RbR', ascending=True)
    # Drop duplicate rows based on ID, keeping only the lowest mean rank
    df = df.drop_duplicates(subset='ID', inplace=False)
    # Normalize the RbR column
    df['RbR'] = (df['RbR'].max() - df['RbR']) / (df['RbR'].max() - df['RbR'].min())
    df = df.rename(columns={'RbR': f'RbR_best_{clustering_metric}'})
    return df[['ID', f'RbR_best_{clustering_metric}']]

def RbR_avg(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
    """
    Calculates the Rank by Rank (RbR) consensus score for each ID in the input dataframe.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the input data with columns 'ID', 'Score1', 'Score2', and so on.
        clustering_metric (str): A string representing the clustering metric used.
        selected_columns (list): A list of strings representing the selected columns for calculating the RbR score.

    Returns:
        pd.DataFrame: A pandas DataFrame with columns 'ID' and 'RbR_{clustering_metric}', where 'RbR_{clustering_metric}' represents the RbR score for each ID.
    """
    # Select the 'ID' column and the selected columns from the input dataframe
    df = df[['ID'] + selected_columns]
    # Convert selected columns to numeric values
    df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors='coerce')
    # Calculate the mean rank across the selected columns
    df['RbR'] = df[selected_columns].mean(axis=1)
    df = df[['ID', 'RbR']]
    # Group the dataframe by 'ID' and calculate the mean of numeric columns
    df = df.groupby('ID', as_index=False).mean(numeric_only=True)
    # Normalize the RbR column
    df['RbR'] = (df['RbR'].max() - df['RbR']) / (df['RbR'].max() - df['RbR'].min())
    df = df.rename(columns={'RbR': f'RbR_avg_{clustering_metric}'})
    return df[['ID', f'RbR_avg_{clustering_metric}']]

def RbV_best(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
    """
    Calculates the Rank by Vote consensus for a given DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing the data.
        clustering_metric (str): A string representing the clustering metric.
        selected_columns (list): A list of column names to consider for the calculation.

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'ID' and 'RbV_' followed by the 'clustering_metric' value.
                     The 'RbV_' column contains the Rank by Vote consensus scores for each ID.
    """
    # Select the 'ID' column and the selected columns from the input dataframe
    df = df[['ID'] + selected_columns]
    # Convert selected columns to numeric values
    df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors='coerce')
    # Initialize a new column 'vote' in the DataFrame with default value 0
    df['RbV'] = 0
    # Increment the f'RbV_{clustering_metric}' column by 1 if the value in each selected column is greater than the 95th percentile of the column
    for column in selected_columns:
        df['RbV'] += (df[column] > df[column].quantile(0.95)).astype(int)
    df = df[['ID', 'RbV']]
    # Sort the DataFrame by 'RbV' in descending order
    df = df.sort_values('RbV', ascending=False)
    # Drop duplicate rows based on ID, keeping only the highest RbV value
    df = df.drop_duplicates('ID', inplace=False)
    # Normalize the RbV column
    df['RbV'] = (df['RbV'] - df['RbV'].min()) / (df['RbV'].max() - df['RbV'].min())
    df = df.rename(columns={'RbV': f'RbV_best_{clustering_metric}'})
    return df[['ID', f'RbV_best_{clustering_metric}']]

def RbV_avg(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
    """
    Calculates the Rank by Vote consensus for a given DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame containing the data.
        clustering_metric (str): A string representing the clustering metric.
        selected_columns (list): A list of column names to consider for the calculation.

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'ID' and 'RbV_' followed by the 'clustering_metric' value.
                     The 'RbV_' column contains the Rank by Vote consensus scores for each ID.
    """
    # Select the 'ID' column and the selected columns from the input dataframe
    df = df[['ID'] + selected_columns]
    # Convert selected columns to numeric values
    df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors='coerce')
    # Initialize a new column 'vote' in the DataFrame with default value 0
    df['RbV'] = 0
    # Increment the f'RbV_{clustering_metric}' column by 1 if the value in each selected column is greater than the 95th percentile of the column
    for column in selected_columns:
        df['RbV'] += (df[column] > df[column].quantile(0.95)).astype(int)
    df = df[['ID', 'RbV']]
    # Group the DataFrame by 'ID' and average across poses
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    # Normalize the RbV column
    df['RbV'] = (df['RbV'] - df['RbV'].min()) / (df['RbV'].max() - df['RbV'].min())
    df = df.rename(columns={'RbV': f'RbV_avg_{clustering_metric}'})
    # Return the DataFrame with columns 'ID' and 'RbV_' followed by the clustering_metric value
    return df[['ID', f'RbV_avg_{clustering_metric}']]

def Zscore_best(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
    '''
    Calculates the Z-score consensus scores for each row in the given DataFrame,
    and aggregates rows by selecting the pose with the best Z-score for each ID.
    
    Args:
    - df: A pandas DataFrame containing the input data with columns 'ID', 'Score1', 'Score2', and so on.
    - clustering_metric: A string representing the clustering metric used.
    - selected_columns: A list of strings representing the selected columns for calculating the Z-score score.
    
    Returns:
    - A pandas DataFrame with columns 'ID' and 'Zscore_best_{clustering_metric}', where 'Zscore_best_{clustering_metric}' represents the Z-score score for each ID. Only the row with the highest Z-score for each ID is included in the output.
    '''
    # Select the 'ID' column and the selected columns from the input dataframe
    df = df[['ID'] + selected_columns]
    # Convert selected columns to numeric values
    df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors='coerce')
    # Calculate Z-scores
    z_scores = (df[selected_columns] - df[selected_columns].mean()) / df[selected_columns].std()
    # Calculate mean Z-score for each row
    df["Zscore"] = z_scores.mean(axis=1)
    df = df[['ID', 'Zscore']]
    # Sort the dataframe by Z-scores in descending order
    df = df.sort_values('Zscore', ascending=False)
    # Drop duplicate rows based on ID, keeping only the highest Z-score
    df = df.drop_duplicates('ID', inplace=False)
    # Normalize the Zscore column
    df['Zscore'] = (df['Zscore'] - df['Zscore'].min()) / (df['Zscore'].max() - df['Zscore'].min())
    df = df.rename(columns={'Zscore': f'Zscore_best_{clustering_metric}'})
    return df[['ID', f'Zscore_best_{clustering_metric}']]

def Zscore_avg(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
    """
    Calculates the Z-score consensus scores for each row in the given DataFrame,
    and aggregates rows by averaging the Z-score for each ID.

    Args:
    - df: A pandas DataFrame containing the input data with columns 'ID', 'Score1', 'Score2', and so on.
    - clustering_metric: A string representing the clustering metric used.
    - selected_columns: A list of strings representing the selected columns for calculating the Z-score score.

    Returns:
    - A pandas DataFrame with columns 'ID' and 'Zscore_avg_{clustering_metric}', where 'Zscore_avg_{clustering_metric}' represents the averaged Z-score for each ID.
    """
    # Select the 'ID' column and the selected columns from the input dataframe
    df = df[['ID'] + selected_columns]
    # Convert selected columns to numeric values
    df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors='coerce')
    # Calculate Z-scores
    z_scores = (df[selected_columns] - df[selected_columns].mean()) / df[selected_columns].std()
    # Average the scores across poses
    df = df.groupby('ID', as_index=False).mean(numeric_only=True)
    # Calculate Z-scores
    z_scores = (df[selected_columns] - df[selected_columns].mean()) / df[selected_columns].std()
    # Calculate mean Z-score for each row
    df["Zscore"] = z_scores.mean(axis=1)
    df = df[['ID', 'Zscore']]
    # Normalize the Zscore column
    df['Zscore'] = (df['Zscore'] - df['Zscore'].min()) / (df['Zscore'].max() - df['Zscore'].min())
    df = df.rename(columns={'Zscore': f'Zscore_avg_{clustering_metric}'})
    # Return DataFrame with 'ID' and 'avg_S_Zscore_{clustering_metric}' columns
    return df[['ID', f'Zscore_avg_{clustering_metric}']]

def ECR_best(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
    """
    Calculates the Exponential Consensus Ranking (ECR) score for each ID in the rescored dataframe and returns the ID for the pose with the best ECR rank.
    
    Args:
        df (pd.DataFrame): A pandas DataFrame containing the rescored data with columns 'ID', 'Score1', 'Score2', and so on.
        clustering_metric (str): A string representing the clustering metric used.
        selected_columns (list): A list of strings representing the selected columns for calculating the ECR score.
    
    Returns:
        pd.DataFrame: A pandas DataFrame with columns 'ID' and 'Method1_ECR_{clustering_metric}', where 'Method1_ECR_{clustering_metric}' represents the ECR score for each ID.
    """
    # Select the 'ID' column and the selected columns from the input dataframe
    df = df[['ID'] + selected_columns]
    # Convert selected columns to numeric values
    df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors='coerce')
    # Calculate the sigma value
    sigma = 0.05 * len(df)
    # Calculate ECR scores for each value in selected columns
    ecr_scores = np.exp(-(df[selected_columns] / sigma))
    # Sum the ECR scores for each ID
    df['ECR'] = ecr_scores[selected_columns].sum(axis=1) / sigma 
    # Drop the selected columns
    df = df[['ID', 'ECR']]
    # Sort by ECR scores in descending order
    df.sort_values('ECR', ascending=False, inplace=True)
    # Drop duplicate rows based on ID, keeping only the highest ECR score
    df.drop_duplicates(subset='ID', inplace=True)
    # Normalize the ECR column
    df['ECR'] = (df['ECR'] - df['ECR'].min()) / (df['ECR'].max() - df['ECR'].min())
    df = df.rename(columns={'ECR': f'ECR_best_{clustering_metric}'})
    # Return dataframe with ID and ECR scores
    return df[['ID', f'ECR_best_{clustering_metric}']]

def ECR_avg(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
    """
    Calculates the Exponential Consensus Ranking (ECR) score for each ID in the input dataframe and returns the ID along with the average ECR rank across the clustered poses.

    Args:
        df (pd.DataFrame): Input dataframe containing the data.
        clustering_metric (str): Clustering metric to be used in the ECR calculation.
        selected_columns (list): List of column names to be used for ECR calculation.

    Returns:
        pd.DataFrame: Dataframe with two columns: 'ID' and `Method2_ECR_{clustering_metric}`. Each row represents an ID and its corresponding average ECR rank across the clustered poses.
    """
    # Select the 'ID' column and the selected columns from the input dataframe
    df = df[['ID'] + selected_columns]
    # Convert selected columns to numeric values
    df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors='coerce')
    # Calculate the sigma value
    sigma = 0.05 * len(df)
    # Calculate the ECR values for each selected column
    ecr_scores = np.exp(-df[selected_columns] / sigma)
    # Sum the ECR values across each row
    df['ECR'] = ecr_scores[selected_columns].sum(axis=1) / sigma
    # Drop the selected columns from the dataframe
    df = df[['ID', 'ECR']]
    # Group the dataframe by 'ID' and calculate the mean of numeric columns
    df = df.groupby('ID', as_index=False).mean(numeric_only=True)
    # Normalize the ECR column
    df['ECR'] = (df['ECR'] - df['ECR'].min()) / (df['ECR'].max() - df['ECR'].min())
    df = df.rename(columns={'ECR': f'ECR_avg_{clustering_metric}'})
    # Return the dataframe with 'ID' and 'ECR_avg_{clustering_metric}' columns
    return df[['ID', f'ECR_avg_{clustering_metric}']]

def avg_ECR(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
    """
    Calculates the Exponential Consensus Ranking (ECR) for a given dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing the data.
        clustering_metric (str): The name of the clustering metric.
        selected_columns (list): The list of selected columns to calculate ECR for.

    Returns:
        pd.DataFrame: The output dataframe with columns 'ID' and 'avg_ECR_clustering' representing the ID and Exponential Consensus Ranking values for the selected columns.
    """
    # Select the 'ID' column and the selected columns from the input dataframe
    df = df[['ID'] + selected_columns]
    # Convert selected columns to numeric values
    df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors='coerce')
    # Calculate the mean ranks for the selected columns
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    # Calculate the sigma value
    sigma = 0.05 * len(df)
    # Calculate the ECR values using the formula
    ecr_scores = np.exp(-df[selected_columns] / sigma)
    # Sum the ECR values across each row
    df['ECR'] = ecr_scores[selected_columns].sum(axis=1) / sigma
    # Normalize the ECR column
    df['ECR'] = (df['ECR'] - df['ECR'].min()) / (df['ECR'].max() - df['ECR'].min())
    df = df.rename(columns={'ECR': f'avg_ECR_{clustering_metric}'})
    return df[['ID', f'avg_ECR_{clustering_metric}']]

def avg_R_ECR(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
    """
    Calculates the Exponential Consensus Ranking (ECR) for a given dataframe.

    Args:
        df (pd.DataFrame): The input dataframe containing the data.
        clustering_metric (str): The name of the clustering metric.
        selected_columns (list): The list of selected columns to calculate ECR for.

    Returns:
        pd.DataFrame: The output dataframe with columns 'ID' and 'avg_ECR_clustering' representing the ID and Exponential Consensus Ranking values for the selected columns.
    """
    # Select the 'ID' column and the selected columns from the input dataframe
    df = df[['ID'] + selected_columns]
    # Convert selected columns to numeric values
    df.loc[:, selected_columns] = df.loc[:, selected_columns].apply(pd.to_numeric, errors='coerce')
    # Calculate the mean ranks for the selected columns
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    # Rerank the mean ranks
    df[selected_columns] = df[selected_columns].rank(method="average", ascending=True, numeric_only=True)
    # Calculate the sigma value
    sigma = 0.05 * len(df)
    # Calculate the ECR values using the formula
    ecr_values = np.exp(-df[selected_columns] / sigma)
    # Sum the ECR values across each row
    df['ECR'] = ecr_values[selected_columns].sum(axis=1) / sigma
    # Normalize the ECR column
    df['ECR'] = (df['ECR'] - df['ECR'].min()) / (df['ECR'].max() - df['ECR'].min())
    df = df.rename(columns={'ECR': f'avg_R_ECR_{clustering_metric}'})
    return df[['ID', f'avg_R_ECR_{clustering_metric}']]

CONSENSUS_METHODS = {
    'avg_ECR': {'function': avg_ECR, 'type': 'rank'},
    'avg_R_ECR': {'function': avg_R_ECR, 'type': 'rank'},
    'ECR_avg': {'function': ECR_avg, 'type': 'rank'},
    'ECR_best': {'function': ECR_best, 'type': 'rank'},
    'RbR_avg': {'function': RbR_avg, 'type': 'rank'},
    'RbR_best': {'function': RbR_best, 'type': 'rank'},
    'RbV_avg': {'function': RbV_avg, 'type': 'score'},
    'RbV_best': {'function': RbV_best, 'type': 'score'},
    'Zscore_avg': {'function': Zscore_avg, 'type': 'score'},
    'Zscore_best': {'function': Zscore_best, 'type': 'score'}
}