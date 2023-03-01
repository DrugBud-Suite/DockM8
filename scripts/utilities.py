import os

def create_temp_folder(path, silent=False):
    if os.path.isdir(path) == True:
        print(f'The folder: {path} already exists')
    else:
        os.mkdir(path)
        if silent == False:
            print(f'The folder: {path} was created')
        
from rdkit.Chem import PandasTools
import multiprocessing
from tqdm import tqdm
        
def split_sdf(dir, sdf_file, ncpus):
    sdf_file_name = os.path.basename(sdf_file).replace('.sdf', '')
    print(f'Splitting SDF file {sdf_file_name}.sdf ...')
    split_files_folder = dir+f'/split_{sdf_file_name}'
    create_temp_folder(dir+f'/split_{sdf_file_name}', silent=True)
    for file in os.listdir(dir+f'/split_{sdf_file_name}'):
        os.unlink(os.path.join(dir+f'/split_{sdf_file_name}', file))
    df = PandasTools.LoadSDF(sdf_file, molColName='Molecule', idName='ID', includeFingerprints=False, strictParsing=True)
    compounds_per_core = round(len(df['ID'])/(ncpus*2))
    used_ids = set() # keep track of used 'ID' values
    file_counter = 1
    for i in tqdm(range(0, len(df), compounds_per_core), desc='Splitting files'):
        chunk = df[i:i+compounds_per_core]
        # remove rows with 'ID' values that have already been used
        chunk = chunk[~chunk['ID'].isin(used_ids)]
        used_ids.update(set(chunk['ID'])) # add new 'ID' values to used_ids
        PandasTools.WriteSDF(chunk, dir+f'/split_{sdf_file_name}/split_' + str(file_counter) + '.sdf', molColName='Molecule', idName='ID')
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

def show_correlation(input, annotation = bool()):
    if isinstance(input, pd.DataFrame):
        dataframe = input
    elif input.endswith('.sdf'):
        dataframe = PandasTools.LoadSDF(input)
    elif input.endswith('.csv'):
        dataframe = pd.read_csv(input, index_col=0)
        dataframe
    matrix = dataframe.corr().round(2)
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    fig, ax = plt.subplots(figsize=(10,10))  
    sns.heatmap(matrix, mask = mask, annot=annotation, vmax=1, vmin=-1, center=0, linewidths=.5, cmap='coolwarm', ax=ax)
    plt.show()

import datetime

def printlog(message):
    def timestamp_generator():
        dateTimeObj = datetime.datetime.now()
        return "["+dateTimeObj.strftime("%Y-%b-%d %H:%M:%S")+"]"
    timestamp = timestamp_generator()
    msg = "\n" + \
        str(timestamp) + \
        ": "+str(message)
    print(msg)
    with open(log_filename, 'a') as f_out:
        f_out.write(msg)

"""
Example usage:
try:
    #something
    text_to_log = "Something done successfully.
    printlog(timestamp_generator(), text_to_log)
except Exception as e:
    text_to_log = f'Something failed due to : {e}'
    printlog(timestamp_generator(), text_to_log)


"""