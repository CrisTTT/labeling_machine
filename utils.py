# utils.py
import numpy as np
import pandas as pd
import warnings

def load_keypoint_data(file_path):
    """
    Loads keypoint data from .npy or .csv files with error handling.

    Args:
        file_path (str): The path to the keypoint file.

    Returns:
        np.ndarray or None: The loaded keypoint data as a NumPy array,
                           or None if loading fails.
    """
    try:
        if file_path.endswith('.npy'):
            print(f"Loading NumPy file: {file_path}")
            keypoints = np.load(file_path)
            print(f"Successfully loaded .npy file. Shape: {keypoints.shape}")
            return keypoints
        elif file_path.endswith('.csv'):
            print(f"Loading CSV file: {file_path}")
            # Assuming CSV contains only keypoint data, possibly with a header.
            # Use header=None if no header is present.
            # Add more robust parsing if structure is complex.
            df = pd.read_csv(file_path)
            print(f"Successfully loaded .csv file. Shape: {df.shape}")
            # Attempt to convert to numpy, assuming numeric data
            keypoints = df.values.astype(float) # Explicitly try float conversion
            print(f"Converted CSV to NumPy array. Shape: {keypoints.shape}")
            return keypoints
        else:
            warnings.warn(f"Unsupported file format: {file_path}. Please use .npy or .csv.", UserWarning)
            return None
    except FileNotFoundError:
        warnings.warn(f"Error: File not found at {file_path}", UserWarning)
        return None
    except ValueError as ve:
         warnings.warn(f"Error loading {file_path}: Possible issue with data format or conversion - {ve}", UserWarning)
         return None
    except pd.errors.EmptyDataError:
        warnings.warn(f"Error: CSV file is empty: {file_path}", UserWarning)
        return None
    except pd.errors.ParserError as pe:
         warnings.warn(f"Error parsing CSV file {file_path}: {pe}", UserWarning)
         return None
    except Exception as e:
        # Catch any other unexpected errors during loading
        warnings.warn(f"An unexpected error occurred while loading {file_path}: {e}", UserWarning)
        return None