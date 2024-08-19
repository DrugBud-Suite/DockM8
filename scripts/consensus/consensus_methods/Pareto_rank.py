import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def is_pareto_efficient_deterministic(costs, return_mask=True):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            ties = np.all(costs[is_efficient] == c, axis=1)
            is_efficient[is_efficient] = is_efficient[is_efficient] | ties
            is_efficient[i] = True
    return is_efficient if return_mask else np.nonzero(is_efficient)[0]

def deterministic_shuffle(df, seed=42):
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

def Pareto_rank_deterministic(df, selected_columns, normalize=True, seed=42):
    df = deterministic_shuffle(df, seed)
    
    df_scores = df[selected_columns].apply(pd.to_numeric, errors="coerce")
    scores = df_scores.values
    
    tie_breaker = np.arange(len(df))
    scores_with_tiebreaker = np.column_stack((scores, tie_breaker))
    
    remaining = np.ones(len(df), dtype=bool)
    ranks = np.zeros(len(df), dtype=int)
    rank = 1

    while np.any(remaining):
        pareto_efficient = is_pareto_efficient_deterministic(scores_with_tiebreaker[remaining])
        ranks[remaining] = np.where(pareto_efficient, rank, ranks[remaining])
        remaining[remaining] = ~pareto_efficient
        rank += 1

    df['Pareto_rank'] = ranks
    
    if normalize:
        df['Pareto_rank'] = (df['Pareto_rank'].max() - df['Pareto_rank']) / (
            df['Pareto_rank'].max() - df['Pareto_rank'].min()
        )
    
    return df[['ID', 'Pareto_rank']]

def Pareto_rank_best(df, selected_columns, normalize=True, seed=42):
    result = Pareto_rank_deterministic(df, selected_columns, normalize=False, seed=seed)
    
    if normalize:
        result['Pareto_rank'] = (result['Pareto_rank'].max() - result['Pareto_rank']) / (
            result['Pareto_rank'].max() - result['Pareto_rank'].min()
        )
    result = result.sort_values("Pareto_rank", ascending=True)
    result = result.drop_duplicates("ID", inplace=False)
    result = result.rename(columns={'Pareto_rank': 'Pareto_rank_best'})
    return result.sort_values('Pareto_rank_best', ascending=True)

def Pareto_rank_avg(df, selected_columns, normalize=True, seed=42):
    result = Pareto_rank_deterministic(df, selected_columns, normalize=False, seed=seed)
    
    if normalize:
        result['Pareto_rank'] = (result['Pareto_rank'].max() - result['Pareto_rank']) / (
            result['Pareto_rank'].max() - result['Pareto_rank'].min()
        )
    result = result.groupby('ID', as_index=False)
    result = result.mean(numeric_only=True)
    result = result.rename(columns={'Pareto_rank': 'Pareto_rank_avg'})
    return result.sort_values('Pareto_rank_avg', ascending=True)