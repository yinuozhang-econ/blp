import numpy as np
import os
import pandas as pd


def get_data(input_name: str): 
    """load data from data folder"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, 'data')
    csv_file = os.path.join(data_folder, input_name+'.csv')
    df = pd.read_csv(csv_file)
    return df