import os 
import sys

import pandas as pd 
pd_idx = pd.IndexSlice

def load_packaging_df(path='../../datasets/AAV2/Fitness_landscape/aav_packaging_all.csv.gz'):
    df = pd.read_csv(path,index_col=list(range(0,11)), header=[0,1,2,3,4,5])
    return df
def load_mouse_df(path='../../datasets/AAV2/Fitness_landscape/aav_mouse_all.csv.gz'):
    return pd.read_csv(path,header=[0,1,2,3,4,5], index_col=list(range(0,11)))
def load_antibody_df(path = '../../datasets/AAV2/Fitness_landscape/aav_antibody_all.csv.gz'):
    return pd.read_csv(path, index_col =list(range(0,11)))
def load_thermostability_df(path='../../datasets/AAV2/Fitness_landscape/aav_thermostability_all.csv.gz'):
    return pd.read_csv(path, header=[0,1,2], index_col=list(range(0,11)))