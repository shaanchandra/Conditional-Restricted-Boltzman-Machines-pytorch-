# Local packages
import os, pickle
from pickletools import TAKEN_FROM_ARGUMENT1
from typing import Union, Tuple, List
import warnings

warnings.filterwarnings("ignore")

# 3rd party modules
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def get_scaler_params(data, scaling_method):
    # Initialize scaler
    if scaling_method == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data)
        params = [scaler.data_min_, scaler.data_max_]
    
    elif scaling_method == "standard":
        scaler = StandardScaler()
        scaler.fit(data)
        params = [scaler.mean_, scaler.var_]
    return scaler, params



def data_preprocess(config, logger, padding_value=-1.0, scaling_method="minmax"):
    """Load the data and preprocess into 2 2d numpy arrays.
        Preprocessing includes:
            1. Normalize data
            2. Create cond and vis variables
        Args:
            - config (dict): the loaded config params
            - scaling_method (str): The scaler method ("standard" or "minmax")
        Returns:
            - vis_mtx (np.array): preprocessed visible data (N-1 x feats)
            - cond_mtx (np.array): preprocessed conditional varibale (data at t-1) (N_combis x cond_feats)
            - scaler_seq (object): the scaling method object for data
            - max_seq_len: the max seq length in the processed data
    """

    #########################
    # Load data
    #########################  


    #########################
    # Scale data
    #########################
    

    ###########################
    # Create vis and cond data
    ###########################
    
    return vis_mtx, cond_mtx, scaler, max_seq_len

