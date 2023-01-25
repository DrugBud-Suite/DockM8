import os

def create_temp_folder(path):
    if os.path.isdir(path) == True:
        print('Temporary folder already exists')
    else:
        os.mkdir(path)
        print('Temporary folder was created')

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