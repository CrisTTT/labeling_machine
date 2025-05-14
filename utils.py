import numpy as np
import pandas as pd
import warnings


def load_keypoint_data(file_path):
    """
    Loads keypoint data from .npy or .csv files with error handling.
    It attempts to return a 3D NumPy array of shape (frames, num_keypoints, 3).
    """
    try:
        if file_path.endswith('.npy'):
            keypoints = np.load(file_path)
            if keypoints.ndim == 3 and keypoints.shape[-1] == 3:
                return keypoints
            elif keypoints.ndim == 2 and keypoints.shape[1] > 0 and keypoints.shape[1] % 3 == 0:
                num_frames = keypoints.shape[0]
                num_points = keypoints.shape[1] // 3
                warnings.warn(
                    f"Warning: Loaded .npy file '{file_path}' was 2D {keypoints.shape}. "
                    f"Reshaping to ({num_frames}, {num_points}, 3).", UserWarning
                )
                return keypoints.reshape(num_frames, num_points, 3)
            else:
                warnings.warn(
                    f"Warning: Loaded .npy file '{file_path}' has an incompatible shape {keypoints.shape}. "
                    "Expected 3D with last dim 3, or 2D with columns multiple of 3.", UserWarning
                )
                return None

        elif file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_path, header=None, skip_blank_lines=True)

                if df.empty:
                    # This case is often caught by EmptyDataError if file is truly empty or unparsable
                    warnings.warn(f"Warning: CSV file '{file_path}' is empty or resulted in an empty DataFrame.",
                                  UserWarning)
                    return None

                df_numeric_coerce = df.apply(pd.to_numeric, errors='coerce')
                df_cleaned_rows = df_numeric_coerce.dropna(axis=0, how='all')

                if df_cleaned_rows.empty:
                    warnings.warn(
                        f"Warning: No data rows found in CSV '{file_path}' after attempting to clean headers/text.",
                        UserWarning)
                    return None

                df_cleaned_cols = df_cleaned_rows.dropna(axis=1, how='all')
                if df_cleaned_cols.empty:
                    warnings.warn(
                        f"Warning: No numeric columns found in CSV '{file_path}' after cleaning text columns.",
                        UserWarning)
                    return None

                potential_keypoint_df = df_cleaned_cols
                num_initial_numeric_cols = df_cleaned_cols.shape[1]

                if num_initial_numeric_cols > 1:
                    first_col_is_int_like = False
                    try:
                        if not df_cleaned_cols.iloc[:, 0].isnull().all():
                            # Check if convertible to float first, then to Int64 for null-safety
                            df_cleaned_cols.iloc[:, 0].astype(float).astype('Int64')
                            first_col_is_int_like = True
                    except (ValueError, TypeError):
                        pass

                    if first_col_is_int_like:
                        cols_after_removing_first = num_initial_numeric_cols - 1
                        if cols_after_removing_first > 0 and cols_after_removing_first % 3 == 0:
                            warnings.warn(
                                f"Info: Assuming the first numeric column in '{file_path}' is a frame index "
                                "and excluding it from keypoint data.", UserWarning
                            )
                            potential_keypoint_df = df_cleaned_cols.iloc[:, 1:]

                if potential_keypoint_df.empty or potential_keypoint_df.shape[1] == 0:
                    warnings.warn(f"Warning: No keypoint data columns remaining in '{file_path}' after processing.",
                                  UserWarning)
                    return None

                # CRITICAL CHECK: After all cleaning, if any NaNs remain in the data intended for keypoints,
                # it means some original non-numeric string data was present where numbers were expected.
                if potential_keypoint_df.isnull().values.any():
                    warnings.warn(
                        f"Warning: CSV file '{file_path}' contains non-numeric values or unparseable entries "
                        "within the presumed keypoint data area. Cannot form clean numeric keypoints.", UserWarning
                    )
                    return None  # Keypoint data must be purely numeric

                keypoints_2d = potential_keypoint_df.values.astype(float)  # Now this should be safe

                num_frames = keypoints_2d.shape[0]
                num_data_columns = keypoints_2d.shape[1]

                if num_data_columns > 0 and num_data_columns % 3 == 0:
                    num_keypoints = num_data_columns // 3
                    keypoints_3d = keypoints_2d.reshape(num_frames, num_keypoints, 3)
                    return keypoints_3d
                else:
                    warnings.warn(
                        f"Warning: Number of final numeric data columns ({num_data_columns}) in CSV '{file_path}' "
                        f"is not divisible by 3. Cannot reshape to (frames, points, 3). "
                        f"Original numeric columns before frame exclusion attempt: {num_initial_numeric_cols}. "
                        f"Processed data shape: {keypoints_2d.shape}", UserWarning
                    )
                    return None

            except pd.errors.EmptyDataError:
                warnings.warn(f"Error: CSV file '{file_path}' is completely empty (EmptyDataError).", UserWarning)
                return None
            except Exception as e:
                warnings.warn(f"Error processing CSV file '{file_path}': {e}", UserWarning)
                return None
        else:
            warnings.warn(f"Unsupported file format: '{file_path}'. Please use .npy or .csv.", UserWarning)
            return None
    except FileNotFoundError:
        warnings.warn(f"Error: File not found at '{file_path}'", UserWarning)
        return None
    except Exception as e:
        warnings.warn(f"An unexpected error occurred while loading '{file_path}': {e}", UserWarning)
        return None
