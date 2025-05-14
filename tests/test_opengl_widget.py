# tests/test_opengl_widget.py
import unittest
import numpy as np

# Add the project root to the Python path to allow importing project modules
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'D:\motion\labeling-machine'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# We need QApplication for Qt signals, even if we don't show the widget
from PyQt6.QtWidgets import QApplication
# Import the class to be tested
from open_gl_widget import OpenGLWidget

# Initialize QApplication once for all tests in this module if needed for signals
# Some Qt functionality might require an application instance.
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)

class TestOpenGLWidget(unittest.TestCase):

    def setUp(self):
        """Create an instance of the widget for each test."""
        # No parent is needed as we are not showing the widget
        self.widget = OpenGLWidget()
        # Create some valid test data
        self.valid_keypoints = np.random.rand(10, 5, 3).astype(np.float32) # 10 frames, 5 points, 3D
        self.valid_limb_seq = [[0, 1], [1, 2], [0, 3], [3, 4]]

        # Create invalid test data
        self.invalid_keypoints_shape = np.random.rand(10, 5, 2).astype(np.float32) # Wrong dimension (2D instead of 3D)
        self.invalid_keypoints_type = [[0, 1, 2], [3, 4, 5]] # Not a numpy array
        self.invalid_limb_seq_format = [[0, 1], [2]] # Incorrect inner list length
        self.invalid_limb_seq_type = "not_a_list"

    def test_initial_state(self):
        """Test the initial state of the widget's attributes."""
        self.assertEqual(self.widget.frame_index, 0)
        self.assertIsNone(self.widget.keypoints)
        self.assertIsNone(self.widget.limbSeq)
        self.assertEqual(self.widget.rotation_x, 0.0)
        self.assertEqual(self.widget.rotation_y, 0.0)
        self.assertEqual(self.widget.zoom_factor, 5.0)

    def test_set_data_valid(self):
        """Test set_data with valid keypoints and limb sequence."""
        self.widget.set_data(self.valid_keypoints, self.valid_limb_seq)
        self.assertIsNotNone(self.widget.keypoints)
        np.testing.assert_array_equal(self.widget.keypoints, self.valid_keypoints)
        self.assertIsNotNone(self.widget.limbSeq)
        self.assertEqual(self.widget.limbSeq, self.valid_limb_seq)

    def test_set_data_invalid_keypoints_shape(self):
        """Test set_data with keypoints of incorrect shape."""
        initial_keypoints = self.widget.keypoints # Should be None initially
        # Use assertWarns or check signal if statusUpdateRequest is connected and mockable
        # For simplicity here, we just check if the attribute remains unchanged (or None)
        self.widget.set_data(self.invalid_keypoints_shape, self.valid_limb_seq)
        # The current implementation prints a warning and sets self.keypoints to None
        self.assertIsNone(self.widget.keypoints, "Keypoints should be None after setting invalid shape.")
        # Check if limbSeq was still set (implementation might allow this)
        # self.assertEqual(self.widget.limbSeq, self.valid_limb_seq) # Or assertIsNone based on desired strictness

    def test_set_data_invalid_keypoints_type(self):
        """Test set_data with keypoints of incorrect type (not ndarray)."""
        self.widget.set_data(self.invalid_keypoints_type, self.valid_limb_seq)
        self.assertIsNone(self.widget.keypoints, "Keypoints should be None after setting invalid type.")

    def test_set_data_invalid_limb_seq_format(self):
        """Test set_data with limb sequence of incorrect format."""
        self.widget.set_data(self.valid_keypoints, self.invalid_limb_seq_format)
        self.assertIsNone(self.widget.limbSeq, "Limb sequence should be None after setting invalid format.")
        # Check if keypoints were still set
        self.assertIsNotNone(self.widget.keypoints, "Keypoints should still be set even if limbSeq is invalid.")
        np.testing.assert_array_equal(self.widget.keypoints, self.valid_keypoints)

    def test_set_data_invalid_limb_seq_type(self):
        """Test set_data with limb sequence of incorrect type."""
        self.widget.set_data(self.valid_keypoints, self.invalid_limb_seq_type)
        self.assertIsNone(self.widget.limbSeq, "Limb sequence should be None after setting invalid type.")
        self.assertIsNotNone(self.widget.keypoints) # Keypoints should still be valid

    def test_set_data_none_inputs(self):
        """Test set_data with None inputs."""
        # First set valid data
        self.widget.set_data(self.valid_keypoints, self.valid_limb_seq)
        self.assertIsNotNone(self.widget.keypoints)
        self.assertIsNotNone(self.widget.limbSeq)
        # Now set None
        self.widget.set_data(None, None)
        self.assertIsNone(self.widget.keypoints)
        self.assertIsNone(self.widget.limbSeq)

    def test_set_frame_index_no_data(self):
        """Test set_frame_index when no keypoint data is loaded."""
        self.widget.set_frame_index(5)
        # frame_index should still update, even if data is None
        self.assertEqual(self.widget.frame_index, 5)

    def test_set_frame_index_with_data_valid(self):
        """Test set_frame_index with valid index when data is loaded."""
        self.widget.set_data(self.valid_keypoints, self.valid_limb_seq) # 10 frames (0-9)
        self.widget.set_frame_index(5)
        self.assertEqual(self.widget.frame_index, 5)
        self.widget.set_frame_index(0)
        self.assertEqual(self.widget.frame_index, 0)
        self.widget.set_frame_index(9)
        self.assertEqual(self.widget.frame_index, 9)

    def test_set_frame_index_with_data_invalid_negative(self):
        """Test set_frame_index with negative index when data is loaded."""
        self.widget.set_data(self.valid_keypoints, self.valid_limb_seq) # 10 frames (0-9)
        # Implementation clamps index to 0..num_frames-1
        self.widget.set_frame_index(-5)
        self.assertEqual(self.widget.frame_index, 0, "Negative index should be clamped to 0.")

    def test_set_frame_index_with_data_invalid_too_high(self):
        """Test set_frame_index with index >= num_frames when data is loaded."""
        self.widget.set_data(self.valid_keypoints, self.valid_limb_seq) # 10 frames (0-9)
        num_frames = self.valid_keypoints.shape[0]
        # Implementation clamps index to 0..num_frames-1
        self.widget.set_frame_index(num_frames) # Index 10 is out of bounds
        self.assertEqual(self.widget.frame_index, num_frames - 1, "Index >= num_frames should be clamped to num_frames-1.")
        self.widget.set_frame_index(num_frames + 100) # Index 110 is out of bounds
        self.assertEqual(self.widget.frame_index, num_frames - 1, "Index > num_frames should be clamped to num_frames-1.")

    # Note: Testing mouse/wheel events requires more advanced setup (e.g., QTest)
    # to simulate events and is omitted here for simplicity.

    # Note: Testing the statusUpdateRequest signal requires mocking or capturing signals,
    # which adds complexity beyond basic attribute checking.

if __name__ == '__main__':
    unittest.main()
