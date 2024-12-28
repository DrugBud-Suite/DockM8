import pandas as pd
import numpy as np

def calculate_zscore(data: pd.DataFrame, columns: list[str], id_column: str = "ID") -> pd.DataFrame:
    """Calculate pure Z-scores."""
    values = data[columns].values
    z_scores = (values - values.mean(axis=0)) / values.std(axis=0)
    zscore_avg = z_scores.mean(axis=1)
    
    return pd.DataFrame({
        id_column: data[id_column],
        'Zscore': zscore_avg
    })

def calculate_rbv(data: pd.DataFrame, columns: list[str], id_column: str = "ID") -> pd.DataFrame:
    """Calculate pure RBV scores."""
    values = data[columns].values
    threshold = np.percentile(values, 95, axis=0)
    rbv_scores = (values > threshold).sum(axis=1)
    
    return pd.DataFrame({
        id_column: data[id_column],
        'RbV': rbv_scores
    })

def rbv_consensus(data: pd.DataFrame, columns: list[str], id_column: str = "ID") -> pd.DataFrame:
    """Combine RBV and Z-score results with proper normalization."""
    # Get individual scores
    rbv_scores = calculate_rbv(data, columns, id_column)
    z_scores = calculate_zscore(data, columns, id_column)
    
    # Merge on ID
    combined = pd.merge(rbv_scores, z_scores, on=id_column, how='inner')
    
    # Normalize z-scores to 0-0.999 range for tiebreaking
    ranks = combined['Zscore'].rank(method='dense') - 1  # -1 to start at 0
    normalized_tiebreaker = ranks / len(ranks)
    
    # Combine scores
    combined['RbV'] = combined['RbV'] + normalized_tiebreaker/5
    
    return combined[[id_column, 'RbV']].sort_values('RbV', ascending=False)
