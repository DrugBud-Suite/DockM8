import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
    # Check if selected columns exist in the dataframe#
    for column in selected_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the dataframe.")
    
    sigma = 0.05 * len(df)
    
    # Calculate ECR scores for each value in selected columns
    try:
        ecr_scores = (np.exp(-(df[selected_columns] / sigma)) / sigma) * 1000
    except KeyError as e:
        raise KeyError(f"Column '{e.args[0]}' not found in the dataframe.")
    
    # Sum the ECR scores for each ID
    df[f'ECR_best_{clustering_metric}'] = ecr_scores.sum(axis=1)
    
    # Drop the selected columns
    df = df[['ID', f'ECR_best_{clustering_metric}']]
    
    # Sort by ECR scores in descending order
    df.sort_values(f'ECR_best_{clustering_metric}', ascending=False, inplace=True)
    
    # Drop duplicate rows based on ID, keeping only the highest ECR score
    df.drop_duplicates(subset='ID', inplace=True)
    
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
    # Calculate the sigma value
    sigma = 0.05 * len(df)
    
    # Calculate the ECR values for each selected column
    ecr_columns = np.exp(-df[selected_columns] / sigma) / sigma * 1000
    
    # Sum the ECR values across each row
    df[f'ECR_avg_{clustering_metric}'] = ecr_columns.sum(axis=1)
    
    # Drop the selected columns from the dataframe
    df = df[['ID', f'ECR_avg_{clustering_metric}']]
    
    # Group the dataframe by 'ID' and calculate the mean of numeric columns
    df2 = df.groupby('ID', as_index=False).mean(numeric_only=True)
    
    # Return the dataframe with 'ID' and 'ECR_avg_{clustering_metric}' columns
    return df2[['ID', f'ECR_avg_{clustering_metric}']]


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
    # Calculate the mean ranks for the selected columns
    mean_ranks = df.groupby('ID', as_index=False)[selected_columns].mean().round(2)
    
    # Rank the mean values in ascending order
    ranks = mean_ranks.rank(method='average', ascending=True)
    
    # Calculate the sigma value
    sigma = 0.05 * len(ranks)
    
    # Calculate the ECR values using the formula
    ecr_values = np.exp(-(ranks / sigma)) / sigma * 1000
    
    # Sum the ECR values across each row
    ecr_sum = ecr_values.sum(axis=1)
    
    # Create a new dataframe with ID and ECR values
    output_df = pd.DataFrame({'ID': mean_ranks['ID'], f'avg_ECR_{clustering_metric}': ecr_sum})
    
    return output_df[['ID', f'avg_ECR_{clustering_metric}']]


def RbR(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
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
    
    # Group the dataframe by 'ID' and calculate the mean of the selected columns
    df = df.groupby('ID', as_index=False).mean(numeric_only=True).round(2)
    
    # Calculate the mean of the selected columns for each row and add it as a new column
    df[f'RbR_{clustering_metric}'] = df[selected_columns].mean(axis=1)

    return df[['ID', f'RbR_{clustering_metric}']]


def RbV(df: pd.DataFrame, clustering_metric: str, selected_columns: list) -> pd.DataFrame:
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
    # Initialize a new column 'vote' in the DataFrame with default value 0
    df[f'RbV_{clustering_metric}'] = 0
    
    # Increment the f'RbV_{clustering_metric}' column by 1 if the value in each selected column is greater than the 95th percentile of the column
    for column in selected_columns:
        df[f'RbV_{clustering_metric}'] += (df[column] > df[column].quantile(0.95)).astype(int)
    
    # Group the DataFrame by 'ID' and calculate the mean of numeric columns
    df = df.groupby('ID', as_index=False).mean(numeric_only=True)
    
    # Round the resulting DataFrame to 2 decimal places
    df = df.round(2)
    
    # Return the DataFrame with columns 'ID' and 'RbV_' followed by the clustering_metric value
    return df[['ID', f'RbV_{clustering_metric}']]


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
    # Convert selected columns to numeric values
    df[selected_columns] = df[selected_columns].apply(pd.to_numeric, errors='coerce')
    
    # Calculate Z-scores
    z_scores = (df[selected_columns] - df[selected_columns].mean()) / df[selected_columns].std()
    
    # Calculate mean Z-score for each row
    consensus_scores = z_scores.mean(axis=1)
    
    # Add new column with mean Z-scores
    df[f'Zscore_best_{clustering_metric}'] = consensus_scores
    
    # Aggregate rows using best Z-score per ID
    df = df.sort_values(f'Zscore_best_{clustering_metric}', ascending=False).drop_duplicates('ID')
    
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
    # Convert selected columns to numeric values
    df[selected_columns] = df[selected_columns].apply(pd.to_numeric, errors='coerce')

    # Calculate Z-scores
    z_scores = (df[selected_columns] - df[selected_columns].mean()) / df[selected_columns].std()

    # Calculate mean Z-score for each row
    consensus_scores = z_scores.mean(axis=1)

    # Add new column with mean Z-scores
    df[f'Zscore_avg_{clustering_metric}'] = consensus_scores

    # Aggregate rows by grouping them by 'ID' and calculating mean of numeric columns
    df = df.groupby('ID', as_index=False).mean(numeric_only=True)

    # Return DataFrame with 'ID' and 'Zscore_avg_{clustering_metric}' columns
    return df[['ID', f'Zscore_avg_{clustering_metric}']]

CONSENSUS_METHODS = {'ECR_best' : {'function' : ECR_best, 'type' : 'rank'},
                     'ECR_avg' : {'function' : ECR_avg, 'type' : 'rank'},
                     'avg_ECR' : {'function' : avg_ECR, 'type' : 'rank'},
                     'RbR' : {'function' : RbR, 'type' : 'rank'},
                     'RbV' : {'function' : RbV, 'type' : 'score'},
                     'Zscore_best' : {'function' : Zscore_best, 'type' : 'score'},
                     'Zscore_avg' : {'function' : Zscore_avg, 'type' : 'score'}}