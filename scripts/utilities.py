import os

def create_temp_folder(path):
    if os.path.isdir(path) == True:
        print(f'The folder: {path} already exists')
    else:
        os.mkdir(path)
        print(f'The folder: {path} was created')
        
from rdkit.Chem import PandasTools
import multiprocessing
        
def split_sdf(w_dir, sdf_file):
    sdf_file_name = os.path.basename(sdf_file).replace('.sdf', '')
    print(f'Splitting SDF file {sdf_file_name}.sdf ...')
    split_files_folder = w_dir+f'/temp/split_{sdf_file_name}'
    create_temp_folder(w_dir+f'/temp/split_{sdf_file_name}')
    for file in os.listdir(w_dir+f'/temp/split_{sdf_file_name}'):
        os.unlink(os.path.join(w_dir+f'/temp/split_{sdf_file_name}', file))
    df = PandasTools.LoadSDF(sdf_file, molColName='Molecule', idName='ID', includeFingerprints=False, strictParsing=True)
    compounds_per_core = round(len(df['ID'])/(multiprocessing.cpu_count()-2))
    used_ids = set() # keep track of used 'ID' values
    file_counter = 1
    for i in range(0, len(df), compounds_per_core):
        chunk = df[i:i+compounds_per_core]
        # remove rows with 'ID' values that have already been used
        chunk = chunk[~chunk['ID'].isin(used_ids)]
        used_ids.update(set(chunk['ID'])) # add new 'ID' values to used_ids
        PandasTools.WriteSDF(chunk, w_dir+f'/temp/split_{sdf_file_name}/split_' + str(file_counter) + '.sdf', molColName='Molecule', idName='ID')
        file_counter+=1
    print(f'Split docking library into {file_counter-1} files each containing {compounds_per_core} compounds')
    return split_files_folder

import pandas as pd

def Insert_row(row_number, df, row_value):
    start_index = 0
    last_index = row_number
    start_lower = row_number
    last_lower = df.shape[0]
    # Create a list of upper_half index and lower half index
    upper_half = [*range(start_index, last_index, 1)]
    lower_half = [*range(start_lower, last_lower, 1)]
    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]  
    # Combine the two lists
    index_ = upper_half + lower_half
    # Update the index of the dataframe
    df.index = index_
    # Insert a row at the end
    df.loc[row_number] = row_value      
    # Sort the index labels
    df = df.sort_index()
    return df

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def show_correlation(input):
    if input.endswith('.sdf'):
        dataframe = PandasTools.LoadSDF(input)
    if input.endswith('.csv'):
        dataframe = pd.read_csv(input)
        dataframe
    if type(input) == pd.DataFrame:
        dataframe = input
    matrix = dataframe.corr().round(2)
    print(matrix)
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, mask = mask, annot=False, vmax=1, vmin=-1, center=0, linewidths=.5, cmap='coolwarm')
    plt.show()