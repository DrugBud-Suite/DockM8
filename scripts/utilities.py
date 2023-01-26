import os

def create_temp_folder(path):
    if os.path.isdir(path) == True:
        print(f'The folder: {path} already exists')
    else:
        os.mkdir(path)
        print(f'The folder: {path} was created')
        
from rdkit.Chem import PandasTools
        
def split_sdf(w_dir, sdf_file, n_compounds):
    create_temp_folder(w_dir+'/temp/split_files')
    for file in os.listdir(w_dir+'/temp/split_files'):
        os.unlink(os.path.join(w_dir+'/temp/split_files', file))
    df = PandasTools.LoadSDF(sdf_file, molColName='Molecule', idName='ID', includeFingerprints=False, strictParsing=True)
    chunks = [df[i:i+n_compounds] for i in range(0, len(df), n_compounds)]
    for i, chunk in enumerate(chunks):
        PandasTools.WriteSDF(chunk, w_dir+'/temp/split_files/split_' + str(i) + '.sdf', molColName='Molecule', idName='ID')

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

def show_correlation(dataframe):
    matrix = dataframe.corr().round(2)
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    sns.heatmap(matrix, mask = mask, annot=True, vmax=1, vmin=-1, center=0, linewidths=.5, cmap='coolwarm')
    plt.show()