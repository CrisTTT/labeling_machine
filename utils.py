import numpy as np
import pandas as pd

def load_keypoint_data(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path).values
    else:
        raise ValueError("Unsupported file format. Please use .npy or .csv.")
