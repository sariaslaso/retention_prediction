import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

def remove_missing(df):
    # df: pandas dataframe
    
    row_count = df.shape[0]
    na_row_count = len(df.loc[df.isnull().any(axis=1)]) # number of rows that contain missing values

    # remove rows with missing values if their count is less than 0.01 percent of the number of rows
    if na_row_count < 0.0001:
        df = df.dropna()

    return df

def remove_duplicates(df):
    # df: pandas dataframe
    
    duplicate_count = df[df.duplicated()].shape[0]
    print(f"The dataset contains {duplicate_count} missing rows.")
    
    return df.drop_duplicates()