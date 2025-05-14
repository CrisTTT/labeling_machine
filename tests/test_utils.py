# tests/test_utils.py
import unittest
import os
import tempfile
import numpy as np
import pandas as pd
import warnings
import re

import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import load_keypoint_data


class TestLoadKeypointData(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # NPY data: 10 frames, 17 keypoints, 3D
        self.npy_data_3d = np.random.rand(10, 17, 3).astype(np.float32)
        self.valid_npy_path = os.path.join(self.temp_dir.name, "valid_keypoints.npy")
        np.save(self.valid_npy_path, self.npy_data_3d)

        # CSV data: Will correspond to the NPY data, but written as 2D
        # (10 * 17 = 170 rows, 3 columns for x,y,z if no frame column)
        # Or (10 rows, 17*3 = 51 columns if frame is handled separately)
        # For this test, we create a CSV that utils.py should reshape
        # It will have 10 frames, and 17*3=51 keypoint columns.
        self.csv_data_frames = 10
        self.csv_data_keypoints_per_frame = 17
        csv_data_flat_coords = self.npy_data_3d.reshape(self.csv_data_frames, self.csv_data_keypoints_per_frame * 3)

        self.valid_csv_path = os.path.join(self.temp_dir.name, "valid_keypoints.csv")
        # Save without a frame column for this specific test of pure keypoint data
        pd.DataFrame(csv_data_flat_coords).to_csv(self.valid_csv_path, index=False, header=False)
        # Expected shape after loading and reshaping by load_keypoint_data
        self.csv_expected_3d_shape = (self.csv_data_frames, self.csv_data_keypoints_per_frame, 3)

        self.empty_csv_with_newline_path = os.path.join(self.temp_dir.name, "empty_with_newline.csv")
        with open(self.empty_csv_with_newline_path, 'w') as f:
            f.write("\n")

        self.truly_empty_csv_path = os.path.join(self.temp_dir.name, "truly_empty.csv")
        with open(self.truly_empty_csv_path, 'w') as f:
            f.write("")

        self.malformed_csv_path = os.path.join(self.temp_dir.name, "malformed.csv")
        # This CSV will have a text value in what should be numeric keypoint data
        malformed_data = np.random.rand(5, 6).astype(str)  # 5 frames, 2 keypoints (6 cols)
        malformed_data[2, 3] = "text_in_data"  # Introduce non-numeric data
        pd.DataFrame(malformed_data).to_csv(self.malformed_csv_path, index=False, header=False)

        self.unsupported_txt_path = os.path.join(self.temp_dir.name, "unsupported.txt")
        with open(self.unsupported_txt_path, 'w') as f: f.write("data")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_valid_npy_file(self):
        loaded_data = load_keypoint_data(self.valid_npy_path)
        self.assertIsNotNone(loaded_data);
        self.assertIsInstance(loaded_data, np.ndarray)
        np.testing.assert_array_almost_equal(loaded_data, self.npy_data_3d, decimal=5)
        self.assertEqual(loaded_data.shape, self.npy_data_3d.shape)

    def test_load_valid_csv_file(self):
        """Test loading a CSV that should be reshaped to 3D by utils.py."""
        loaded_data = load_keypoint_data(self.valid_csv_path)
        self.assertIsNotNone(loaded_data, "Data should be loaded from valid CSV.")
        self.assertIsInstance(loaded_data, np.ndarray, "Loaded CSV data should be a NumPy array.")
        self.assertEqual(loaded_data.shape, self.csv_expected_3d_shape,
                         f"Loaded CSV data shape mismatch. Expected {self.csv_expected_3d_shape}, got {loaded_data.shape}")
        # Compare content by reshaping the original npy_data_3d to match how it would be loaded
        np.testing.assert_array_almost_equal(loaded_data, self.npy_data_3d, decimal=5)

    def test_load_non_existent_file(self):
        non_existent_path = os.path.join(self.temp_dir.name, "does_not_exist.npy")
        with self.assertWarnsRegex(UserWarning, "Error: File not found"):
            loaded_data = load_keypoint_data(non_existent_path)
        self.assertIsNone(loaded_data)

    def test_load_unsupported_file_format(self):
        with self.assertWarnsRegex(UserWarning,
                                   "Unsupported file format:.*unsupported.txt.*Please use \\.npy or \\.csv"):
            loaded_data = load_keypoint_data(self.unsupported_txt_path)
        self.assertIsNone(loaded_data)

    def test_load_empty_csv_file_with_newline(self):
        # This should be caught by "Warning: CSV file '...' is empty or contains only blank lines."
        # OR "Error: CSV file '...' is completely empty (EmptyDataError)."
        expected_warning_pattern = (
            r"Warning: CSV file '.*empty_with_newline\.csv' is empty or contains only blank lines\.|"
            r"Error: CSV file '.*empty_with_newline\.csv' is completely empty \(EmptyDataError\)\."
        )
        with self.assertWarnsRegex(UserWarning, expected_warning_pattern):
            loaded_data = load_keypoint_data(self.empty_csv_with_newline_path)
        self.assertIsNone(loaded_data)

    def test_load_truly_empty_csv_file(self):
        # This should be caught by EmptyDataError or the df.empty check
        expected_warning_pattern = (
            r"Error: CSV file '.*truly_empty\.csv' is completely empty \(EmptyDataError\)\.|"
            r"Warning: CSV file '.*truly_empty\.csv' is empty or contains only blank lines\."
        )
        with self.assertWarnsRegex(UserWarning, expected_warning_pattern):
            loaded_data = load_keypoint_data(self.truly_empty_csv_path)
        self.assertIsNone(loaded_data)

    def test_load_malformed_csv_file(self):
        """Test CSV with non-numeric data in keypoint area."""
        # Expecting the warning about NaNs in the presumed keypoint data area
        expected_warning_pattern = (
            r"Warning: CSV file '.*malformed\.csv' contains non-numeric values or unparseable entries "
            r"within the presumed keypoint data area\. Cannot form clean numeric keypoints\."
        )
        with self.assertWarnsRegex(UserWarning, expected_warning_pattern):
            loaded_data = load_keypoint_data(self.malformed_csv_path)
        self.assertIsNone(loaded_data, "Should return None for malformed CSV with text in data.")

    def test_load_directory_as_file(self):
        normalized_path_regex = re.escape(self.temp_dir.name).replace(re.escape(os.sep), r"[\\/]+")
        # Expect the full warning message including the suffix
        expected_warning_regex = f"Unsupported file format: '{normalized_path_regex}'\\. Please use \\.npy or \\.csv\\."
        with self.assertWarnsRegex(UserWarning, expected_warning_regex):
            loaded_data = load_keypoint_data(self.temp_dir.name)
        self.assertIsNone(loaded_data)


if __name__ == '__main__':
    unittest.main()
