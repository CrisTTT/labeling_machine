import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import numpy as np
import pandas as pd
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'D:\motion\labeling-machine'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt6.QtWidgets import QApplication, QLineEdit, QCheckBox
from interface import Interface, \
    LabelUIElements  # Assuming LabelUIElements is accessible or part of Interface

# Ensure a QApplication instance exists for Qt-dependent parts
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


class TestInterfaceDataHandling(unittest.TestCase):

    def setUp(self):
        """Set up a Interface instance and temporary files for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create an Interface instance (it will create its own UI elements)
        # We won't show it, just test its logic.
        self.interface = Interface()

        # Mock essential UI elements that are directly accessed or would be created
        # by methods we are not calling directly (like _add_label_ui if not triggered)
        # For methods like _load_or_initialize_label_data, it expects label_names
        # and label_ui_elements to be somewhat populated.

        # Mock keypoints data
        self.num_frames = 5
        self.num_keypoints = 3
        self.interface.keypoints = np.random.rand(self.num_frames, self.num_keypoints, 3).astype(np.float32)
        self.interface.keypoints_path = os.path.join(self.temp_dir.name, "dummy_keypoints.npy")  # Needs a path

        # Define some label names and their types for tests
        self.interface.label_names = ["action", "object_visible", "count"]
        self.interface.label_is_numeric = {
            "action": False,
            "object_visible": False,  # Will treat as boolean-like string "True"/"False"
            "count": True
        }
        # Mock UI elements that _load_or_initialize_label_data might interact with if it read numeric status from them
        # Or ensure label_is_numeric is the source of truth for the tests
        self.interface.label_ui_elements = {
            "action": MagicMock(spec=LabelUIElements, name_widget=MagicMock(), value_widget=MagicMock(spec=QLineEdit),
                                numeric_checkbox=MagicMock(spec=QCheckBox, isChecked=lambda: False)),
            "object_visible": MagicMock(spec=LabelUIElements, name_widget=MagicMock(),
                                        value_widget=MagicMock(spec=QLineEdit),
                                        numeric_checkbox=MagicMock(spec=QCheckBox, isChecked=lambda: False)),
            "count": MagicMock(spec=LabelUIElements, name_widget=MagicMock(), value_widget=MagicMock(spec=QLineEdit),
                               numeric_checkbox=MagicMock(spec=QCheckBox, isChecked=lambda: True))
        }

        # Mock climber and route ID inputs
        self.interface.climber_id_input = MagicMock(spec=QLineEdit)
        self.interface.climber_id_input.text.return_value = "test_climber"
        self.interface.route_id_input = MagicMock(spec=QLineEdit)
        self.interface.route_id_input.text.return_value = "test_route"

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
        # Explicitly delete interface to help with Qt resource cleanup if any test fails mid-way
        del self.interface

    def test_initialize_label_data_no_existing_csv(self):
        """Test _load_or_initialize_label_data when no CSV file exists."""
        # Ensure csv_file path is set as it would be in the app
        self.interface.csv_file = os.path.join(self.temp_dir.name, "non_existent_labels.csv")

        self.interface._load_or_initialize_label_data()

        self.assertIsNotNone(self.interface.label_values, "label_values should be initialized.")
        self.assertEqual(len(self.interface.label_values), self.num_frames, "Should have entries for all frames.")
        for frame_idx in range(self.num_frames):
            self.assertIn(frame_idx, self.interface.label_values)
            frame_data = self.interface.label_values[frame_idx]
            self.assertEqual(frame_data["action"], "", "Default for non-numeric should be empty string.")
            self.assertEqual(frame_data["object_visible"], "", "Default for non-numeric should be empty string.")
            self.assertEqual(frame_data["count"], 0.0, "Default for numeric should be 0.0.")

    def test_update_label_data_from_input_non_numeric(self):
        """Test _update_label_data_from_input for a non-numeric label."""
        self.interface.frame_index = 0
        # Initialize label_values for frame 0 if not already
        if 0 not in self.interface.label_values:
            self.interface.label_values[0] = {name: "" for name in self.interface.label_names}
            self.interface.label_values[0]["count"] = 0.0

        mock_input_widget = MagicMock(spec=QLineEdit)
        mock_checkbox = MagicMock(spec=QCheckBox)
        mock_checkbox.isChecked.return_value = False  # Non-numeric

        self.interface._update_label_data_from_input("action", "test_action", mock_input_widget, mock_checkbox)
        self.assertEqual(self.interface.label_values[0]["action"], "test_action")
        self.assertTrue(self.interface._has_unsaved_changes)

    def test_update_label_data_from_input_numeric_valid(self):
        """Test _update_label_data_from_input for a numeric label with valid input."""
        self.interface.frame_index = 1
        if 1 not in self.interface.label_values:  # Initialize if needed for test
            self.interface.label_values[1] = {name: "" for name in self.interface.label_names}
            self.interface.label_values[1]["count"] = 0.0

        mock_input_widget = MagicMock(spec=QLineEdit)
        mock_checkbox = MagicMock(spec=QCheckBox)
        mock_checkbox.isChecked.return_value = True  # Numeric

        self.interface._update_label_data_from_input("count", "123.45", mock_input_widget, mock_checkbox)
        self.assertEqual(self.interface.label_values[1]["count"], 123.45)
        mock_input_widget.setStyleSheet.assert_called_with("")  # Check style reset

    def test_update_label_data_from_input_numeric_invalid(self):
        """Test _update_label_data_from_input for a numeric label with invalid input."""
        self.interface.frame_index = 2
        if 2 not in self.interface.label_values:  # Initialize if needed for test
            self.interface.label_values[2] = {name: "" for name in self.interface.label_names}
            self.interface.label_values[2]["count"] = 5.0  # Pre-existing value

        mock_input_widget = MagicMock(spec=QLineEdit)
        mock_checkbox = MagicMock(spec=QCheckBox)
        mock_checkbox.isChecked.return_value = True  # Numeric

        self.interface._update_label_data_from_input("count", "invalid_float", mock_input_widget, mock_checkbox)
        # Current implementation defaults to 0.0 on parse error
        self.assertEqual(self.interface.label_values[2]["count"], 0.0)
        mock_input_widget.setStyleSheet.assert_called_with("QLineEdit { background-color: #ffdddd; }")

    @patch('interface.QFileDialog.getSaveFileName')
    def test_save_csv_data_preparation(self, mock_get_save_file_name):
        """Test the data structure prepared by save_csv before writing."""
        # Mock QFileDialog to return a dummy path and not show a dialog
        mock_get_save_file_name.return_value = (
        os.path.join(self.temp_dir.name, "test_output.csv"), "CSV Files (*.csv)")

        # Populate label_values with some data
        self.interface.label_values = {}
        for i in range(self.num_frames):
            self.interface.label_values[i] = {
                "action": f"walk_{i}",
                "object_visible": "True",
                "count": float(i)
            }

        # Mock the part of save_csv that creates the DataFrame
        # We need to capture the 'data_to_save' list
        with patch('interface.pd.DataFrame') as mock_pd_dataframe:
            self.interface.save_csv(show_dialog=False)  # Use current self.csv_file or trigger internal logic

            self.assertTrue(mock_pd_dataframe.called, "pandas.DataFrame should have been called.")

            # Get the arguments passed to pd.DataFrame
            call_args = mock_pd_dataframe.call_args
            data_saved_list = call_args[0][0]  # First positional argument is the data list
            columns_saved = call_args[1]['columns']  # Keyword argument 'columns'

            self.assertEqual(len(data_saved_list), self.num_frames)

            expected_headers = ["climber_id", "route_id", "frame"]
            for kp_idx in range(self.num_keypoints):
                expected_headers.extend([f"kp{kp_idx}_x", f"kp{kp_idx}_y", f"kp{kp_idx}_z"])
            expected_headers.extend(self.interface.label_names)
            self.assertEqual(columns_saved, expected_headers)

            for i in range(self.num_frames):
                row = data_saved_list[i]
                self.assertEqual(row["frame"], i)
                self.assertEqual(row["action"], f"walk_{i}")
                self.assertEqual(row["object_visible"], "True")
                self.assertEqual(row["count"], float(i))
                self.assertEqual(row["climber_id"], "test_climber")


if __name__ == '__main__':
    unittest.main()
