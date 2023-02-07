from rdkit.Chem import PandasTools
import pandas as pd

def calculate_EFs(w_dir, docking_library):
    original_df = PandasTools.LoadSDF(docking_library, molColName='Molecule', idName='ID')
    original_df = original_df[['ID', 'Activity']]
    ranking_results = pd.read_csv(w_dir+'/temp/ranking/method_results.csv')
    merged_df = pd.merge(ranking_results, original_df, on='ID')
    merged_df['Activity'] = merged_df['Activity'].apply(pd.to_numeric)
    method_list = ranking_results.columns.values.tolist()
    method_list = method_list[1:]
    EF_results = pd.DataFrame()
    for x in method_list:
        sorted_df = merged_df.sort_values(x, ascending = False)
        percent10 = 0.10 * len(sorted_df)
        percent10_df = sorted_df.head(round(percent10))
        ef10 = (percent10_df['Activity'].sum())/len(percent10_df)*100
        percent1 = 0.01 * len(sorted_df)
        percent1_df = sorted_df.head(round(percent1))
        ef1 = (percent1_df['Activity'].sum())/len(percent1_df)*100
        EF_results.loc[x, 'EF10%'] = ef10
        EF_results.loc[x, 'EF1%'] = ef1
    EF_results.to_csv(w_dir+'/temp/ranking/enrichement_factors.csv')
    
def calculate_EFs_simplified(w_dir, docking_library):
    original_df = PandasTools.LoadSDF(docking_library, molColName='Molecule', idName='ID')
    original_df = original_df[['ID', 'Activity']]
    ranking_results = pd.read_csv(w_dir+'/temp/ranking/method_results.csv')
    merged_df = ranking_results.merge(original_df, on='ID')
    merged_df['Activity'] = pd.to_numeric(merged_df['Activity'])
    method_list = ranking_results.columns.tolist()[1:]
    EF_results = pd.DataFrame()
    for method in method_list:
        sorted_df = merged_df.sort_values(method, ascending = False)
        ef10 = (sorted_df.head(len(sorted_df)//10)['Activity'].sum())/len(sorted_df.head(len(sorted_df)//10))*100
        ef1 = (sorted_df.head(len(sorted_df)//100)['Activity'].sum())/len(sorted_df.head(len(sorted_df)//100))*100
        EF_results.loc[method, 'EF10%'] = ef10
        EF_results.loc[method, 'EF1%'] = ef1
    EF_results.to_csv(w_dir+'/temp/ranking/enrichement_factors.csv')