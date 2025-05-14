# tests/test_classic_interface.py
import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import numpy as np
import pandas as pd

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt6.QtWidgets import QApplication, QLineEdit, QCheckBox, QToolButton, QVBoxLayout, QWidget
from classic_interface import ClassicInterface, LabelData

app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

class TestClassicInterfaceDataHandling(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.interface = ClassicInterface()
        self.num_frames = 5; self.num_keypoints = 3
        self.interface.keypoints = np.random.rand(self.num_frames, self.num_keypoints, 3).astype(np.float32)
        self.interface.keypoints_path = os.path.join(self.temp_dir.name, "dummy_keypoints.npy")
        self.interface.climber_id_input = MagicMock(spec=QLineEdit); self.interface.climber_id_input.text.return_value = "classic_climber"
        self.interface.route_id_input = MagicMock(spec=QLineEdit); self.interface.route_id_input.text.return_value = "classic_route"
        self.interface.labels_layout = MagicMock(spec=QVBoxLayout)
        self.interface.labels_widget = MagicMock(spec=QWidget)

    def tearDown(self):
        self.temp_dir.cleanup(); del self.interface

    def test_add_label_type(self):
        self.assertEqual(len(self.interface.label_data_list), 0)
        with patch.object(self.interface, '_create_label_ui', return_value=MagicMock()) as mock_create_ui:
            self.interface.add_label_type(name="activity", is_numeric=False)
        self.assertEqual(len(self.interface.label_data_list), 1); mock_create_ui.assert_called_once()
        label_data = self.interface.label_data_list[0]
        self.assertIsInstance(label_data, LabelData); self.assertEqual(label_data.name, "activity")
        self.assertFalse(label_data.is_numeric); self.assertEqual(len(label_data.values), 0)

    def test_add_value_category_to_label(self):
        with patch.object(self.interface, '_create_label_ui', return_value=MagicMock()):
            self.interface.add_label_type(name="behavior")
        label_data = self.interface.label_data_list[0]
        mock_values_container_layout = MagicMock(spec=QVBoxLayout)
        # Ensure _create_value_ui is called by _add_value_category
        with patch.object(self.interface, '_create_value_ui') as mock_create_value_ui:
            self.interface._add_value_category(label_data, mock_values_container_layout)
        mock_create_value_ui.assert_called_once() # Check it was called
        self.assertEqual(len(label_data.values), 1)
        value_entry = label_data.values[0]
        self.assertEqual(value_entry[0], ""); self.assertEqual(len(value_entry[1]), 0)

    def test_add_interval_to_value_category(self):
        with patch.object(self.interface, '_create_label_ui', return_value=MagicMock()):
            self.interface.add_label_type(name="state")
        label_data = self.interface.label_data_list[0]
        mock_values_container_layout = MagicMock(spec=QVBoxLayout)
        with patch.object(self.interface, '_create_value_ui'):
             self.interface._add_value_category(label_data, mock_values_container_layout)
        value_str, intervals_list_ref = label_data.values[0]
        mock_intervals_container_layout = MagicMock(spec=QVBoxLayout)
        # Ensure _create_interval_ui is called by _add_interval
        with patch.object(self.interface, '_create_interval_ui') as mock_create_interval_ui:
            self.interface._add_interval(label_data, intervals_list_ref, mock_intervals_container_layout)
        mock_create_interval_ui.assert_called_once() # Check it was called
        self.assertEqual(len(intervals_list_ref), 1)
        interval_pair = intervals_list_ref[0]
        self.assertEqual(interval_pair[0], 0); self.assertEqual(interval_pair[1], 0)

    def test_update_internal_data_structures(self):
        with patch.object(self.interface, '_create_label_ui', return_value=MagicMock()):
            self.interface.add_label_type(name="test_label", is_numeric=True)
        label_data = self.interface.label_data_list[0]; self.assertEqual(label_data.name, "test_label")
        value_name = "category1"; intervals_for_category1 = []; label_data.values.append((value_name, intervals_for_category1))
        self.assertEqual(label_data.values[0][0], "category1")
        start_frame, end_frame = 10, 20; intervals_for_category1.append([start_frame, end_frame])
        self.assertEqual(intervals_for_category1[0], [10, 20])
        self.interface._update_interval_frame(intervals_for_category1[0], 0, "15")
        self.interface._update_interval_frame(intervals_for_category1[0], 1, "25")
        self.assertEqual(intervals_for_category1[0], [15, 25])
        self.interface._update_value_string(label_data, 0, "category1_updated")
        self.assertEqual(label_data.values[0][0], "category1_updated")

    @patch('classic_interface.QFileDialog.getSaveFileName')
    def test_save_csv_data_preparation_logic(self, mock_get_save_file_name):
        mock_get_save_file_name.return_value = (os.path.join(self.temp_dir.name, "classic_output.csv"), "CSV Files (*.csv)")
        action_label = LabelData(name="action", is_numeric=False); action_intervals_run = [[0, 2]]; action_intervals_jump = [[3, 3]]
        action_label.values.append(("run", action_intervals_run)); action_label.values.append(("jump", action_intervals_jump))
        self.interface.label_data_list.append(action_label)
        height_label = LabelData(name="height", is_numeric=True); height_intervals_low = [[0, 1]]; height_intervals_high = [[2, 4]]
        height_label.values.append(("1.5", height_intervals_low)); height_label.values.append(("3.0", height_intervals_high))
        self.interface.label_data_list.append(height_label)
        empty_label_data = LabelData(name="empty_label", is_numeric=False); self.interface.label_data_list.append(empty_label_data)
        with patch('classic_interface.pd.DataFrame') as mock_pd_dataframe:
            self.interface.save_csv()
            self.assertTrue(mock_pd_dataframe.called); call_args = mock_pd_dataframe.call_args
            data_saved_list = call_args[0][0]; columns_saved = call_args[1]['columns']
            self.assertEqual(len(data_saved_list), self.num_frames)
            expected_headers = ["climber_id", "route_id", "frame"]
            for kp_idx in range(self.num_keypoints): expected_headers.extend([f"kp{kp_idx}_x", f"kp{kp_idx}_y", f"kp{kp_idx}_z"])
            expected_headers.extend(["action", "height", "empty_label"]); self.assertEqual(columns_saved, expected_headers)
            self.assertEqual(data_saved_list[0]["action"], "run"); self.assertEqual(data_saved_list[0]["height"], 1.5); self.assertEqual(data_saved_list[0]["empty_label"], "")
            self.assertEqual(data_saved_list[1]["action"], "run"); self.assertEqual(data_saved_list[1]["height"], 1.5); self.assertEqual(data_saved_list[1]["empty_label"], "")
            self.assertEqual(data_saved_list[2]["action"], "run"); self.assertEqual(data_saved_list[2]["height"], 3.0); self.assertEqual(data_saved_list[2]["empty_label"], "")
            self.assertEqual(data_saved_list[3]["action"], "jump"); self.assertEqual(data_saved_list[3]["height"], 3.0); self.assertEqual(data_saved_list[3]["empty_label"], "")
            self.assertEqual(data_saved_list[4]["action"], ""); self.assertEqual(data_saved_list[4]["height"], 3.0); self.assertEqual(data_saved_list[4]["empty_label"], "")

    def test_save_csv_no_labels_defined(self):
        self.interface.label_data_list = []
        with patch('classic_interface.QMessageBox.critical') as mock_msg_box, \
             patch('classic_interface.pd.DataFrame') as mock_pd_dataframe:
            self.interface.save_csv()
            mock_msg_box.assert_called_once()
            # FIX: Check the message (args[0][2]) not the title (args[0][1])
            self.assertIn("no labels defined or named", mock_msg_box.call_args[0][2].lower())
            mock_pd_dataframe.assert_not_called()

    def test_save_csv_label_with_empty_name(self):
        unnamed_label = LabelData(name="", is_numeric=False)
        named_label = LabelData(name="valid_label", is_numeric=False); named_label.values.append(("val1", [[0,0]]))
        self.interface.label_data_list = [unnamed_label, named_label]
        with patch('classic_interface.QFileDialog.getSaveFileName') as mock_dialog, \
             patch('classic_interface.pd.DataFrame') as mock_pd_dataframe, \
             patch('warnings.warn') as mock_warn:
            mock_dialog.return_value = (os.path.join(self.temp_dir.name, "output.csv"), "CSV")
            self.interface.save_csv()
            mock_warn.assert_any_call("Skipping label type with empty name during save.", UserWarning)
            self.assertTrue(mock_pd_dataframe.called)
            columns_saved = mock_pd_dataframe.call_args[1]['columns']
            self.assertIn("valid_label", columns_saved); self.assertNotIn("", columns_saved)

if __name__ == '__main__':
    unittest.main()
