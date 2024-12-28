import pandas as pd
import numpy as np

def soft_rbv_consensus(data: pd.DataFrame, columns: list[str], id_column: str = "ID",
                      threshold_percentile: float = 95,
                      sigma: float = 0.1) -> pd.DataFrame:
    """
    Calculate RBV consensus score using soft voting.
    
    Instead of binary votes, this implementation uses a continuous voting scheme
    where each score contributes a weighted vote based on its distance from the
    threshold. The weighting uses a sigmoidal function to create a smooth
    transition around the threshold.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data containing scoring columns
    columns : list[str]
        Names of scoring function columns
    id_column : str
        Name of the ID column
    threshold_percentile : float
        Percentile to use for threshold calculation (default: 95)
    sigma : float
        Controls the steepness of the sigmoid function (default: 0.1)
        Smaller values create sharper transitions
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with ID and soft RBV score
    """
    values = data[columns].values
    
    # Calculate thresholds for each scoring function
    thresholds = np.percentile(values, threshold_percentile, axis=0)
    
    # Calculate scale factors for normalization
    # This helps make different scoring functions more comparable
    scales = values.std(axis=0)
    
    # Calculate normalized distances from threshold
    normalized_distances = (values - thresholds) / scales
    
    # Apply sigmoid function to get soft votes
    # sigmoid(x) = 1 / (1 + exp(-x/sigma))
    soft_votes = 1 / (1 + np.exp(-normalized_distances/sigma))
    
    # Sum the soft votes to get final score
    soft_rbv_scores = soft_votes.sum(axis=1)
    
    return pd.DataFrame({
        id_column: data[id_column],
        'SoftRbV': soft_rbv_scores
    }).sort_values('SoftRbV', ascending=False)
