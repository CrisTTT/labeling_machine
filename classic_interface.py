# classic_interface.py
# (Same imports as before, including QApplication)
import sys
import os
import numpy as np
import cv2
import pandas as pd
from PyQt6.QtWidgets import (QMainWindow, QHBoxLayout, QVBoxLayout, QLabel,
                             QSlider, QWidget, QPushButton, QFileDialog,
                             QLineEdit, QFormLayout, QCheckBox, QToolButton,
                             QScrollArea, QMessageBox, QSizePolicy, QStatusBar,
                             QApplication) # Ensure QApplication is imported
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon, QCloseEvent

from open_gl_widget import OpenGLWidget
from utils import load_keypoint_data
import warnings

# (LabelData class remains the same)
class LabelData:
    """ Helper class to store data for a single label category. """
    def __init__(self, name="", is_numeric=False):
        self.name = name
        self.is_numeric = is_numeric
        self.values = [] # Format: [ (value_str, [ [start,end], [start,end] ]), ... ]


class ClassicInterface(QMainWindow):
    # ( __init__ remains the same)
    def __init__(self):
        super().__init__()
        # Data
        self.keypoints = None # numpy array (frames, points, 3)
        self.video_path = None
        self.keypoints_path = None
        self.cap = None
        self.total_frames = 0
        self.frame_index = 0
        # [[0, 1], [1, 2], ...]
        self.limbSeq = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],
                        [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12],
                        [12, 13], [14, 15], [15, 16]]
        self.label_data_list = [] # List of LabelData objects
        self.ui_widgets = {} # Maps LabelData object to its UI widgets {label_data: {widget_name: widget_ref}}

        self.initUI()
        self.update_widget_states() # Initial state (most things disabled)

    # (initUI remains the same)
    def initUI(self):
        self.setWindowTitle('Classic 3D Pose Labeling Tool')
        self.setGeometry(100, 100, 1300, 700) # Increased size slightly
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready. Load video and keypoints to begin.")
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        left_panel = QVBoxLayout()
        self.video_label = QLabel("No Video Loaded", self); self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.video_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); self.video_label.setMinimumSize(480, 270)
        left_panel.addWidget(self.video_label, stretch=1)
        self.openGLWidget = OpenGLWidget(self); self.openGLWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); self.openGLWidget.setMinimumSize(480, 270)
        self.openGLWidget.statusUpdateRequest.connect(self.show_status_message)
        left_panel.addWidget(self.openGLWidget, stretch=1)
        main_layout.addLayout(left_panel, stretch=2)
        right_panel = QVBoxLayout(); right_panel.setSpacing(10)
        file_layout = QFormLayout()
        self.load_video_button = QPushButton("Load Video (.mp4)", self); self.load_video_button.clicked.connect(self.load_video)
        self.load_keypoints_button = QPushButton("Load Keypoints (.npy, .csv)", self); self.load_keypoints_button.clicked.connect(self.load_keypoints)
        self.video_path_label = QLabel("None", self); self.keypoints_path_label = QLabel("None", self)
        file_layout.addRow(self.load_video_button, self.video_path_label); file_layout.addRow(self.load_keypoints_button, self.keypoints_path_label)
        right_panel.addLayout(file_layout)
        nav_layout = QHBoxLayout()
        self.frame_label = QLabel(f"Frame: {self.frame_index} / {self.total_frames -1}", self)
        self.slider = QSlider(Qt.Orientation.Horizontal, self); self.slider.setMinimum(0); self.slider.setMaximum(0); self.slider.valueChanged.connect(self.slider_update_frame); self.slider.setEnabled(False)
        nav_layout.addWidget(self.frame_label); nav_layout.addWidget(self.slider, stretch=1)
        right_panel.addLayout(nav_layout)
        id_layout = QFormLayout()
        self.climber_id_input = QLineEdit("climber_001", self); self.route_id_input = QLineEdit("route_001", self)
        id_layout.addRow("Climber ID:", self.climber_id_input); id_layout.addRow("Route ID:", self.route_id_input)
        right_panel.addLayout(id_layout)
        label_controls_layout = QHBoxLayout()
        self.load_labels_button = QPushButton("Load Label Names (.txt)", self); self.load_labels_button.clicked.connect(self.load_label_names)
        self.add_label_button = QPushButton("Add New Label Type", self); self.add_label_button.clicked.connect(self.add_new_label_type)
        self.save_button = QPushButton("Save Labels to CSV", self); self.save_button.clicked.connect(self.save_csv)
        label_controls_layout.addWidget(self.load_labels_button); label_controls_layout.addWidget(self.add_label_button); label_controls_layout.addWidget(self.save_button)
        right_panel.addLayout(label_controls_layout)
        right_panel.addWidget(QLabel("--- Labels ---"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.labels_scroll_area = QScrollArea(self); self.labels_scroll_area.setWidgetResizable(True); self.labels_scroll_area.setStyleSheet("background-color: #ffffff;")
        self.labels_widget = QWidget(); self.labels_layout = QVBoxLayout(self.labels_widget); self.labels_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.labels_scroll_area.setWidget(self.labels_widget)
        right_panel.addWidget(self.labels_scroll_area, stretch=1)
        main_layout.addLayout(right_panel, stretch=1)
        self.show()

    # --- UI Update ---

    def update_widget_states(self):
        """ Enable/disable widgets based on loaded data. """
        # print("DEBUG: Entering update_widget_states") # Keep debug print if needed
        has_video = self.cap is not None and self.cap.isOpened()
        has_keypoints = self.keypoints is not None

        self.slider.setEnabled(has_video or has_keypoints)
        self.openGLWidget.setEnabled(has_keypoints)

        # --- FIX: Cast to bool ---
        self.save_button.setEnabled(bool(has_keypoints and self.label_data_list))

        self.add_label_button.setEnabled(True)
        self.load_labels_button.setEnabled(True)

        if has_video or has_keypoints:
             # Ensure total_frames is calculated before setting max
             if hasattr(self, 'total_frames'):
                self.slider.setMaximum(self.total_frames - 1 if self.total_frames > 0 else 0)
             else:
                self.slider.setMaximum(0) # Should not happen after init normally
        else:
             self.slider.setMaximum(0)
             self.frame_index = 0

        # Avoid calling update_frame_display from here if it causes issues?
        # Or ensure update_frame_display is robust. Let's keep it for now.
        self.update_frame_display()
        # print("DEBUG: Exiting update_widget_states")


    # (show_status_message, display_error_message remain the same)
    def show_status_message(self, message, timeout=3000):
        self.statusBar.showMessage(message, timeout)

    def display_error_message(self, title, message):
        print(f"DEBUG: Error message: {title} - {message}") # Debug print
        QMessageBox.critical(self, title, message)
        self.show_status_message(f"Error: {message}", 5000)


    # (load_video remains the same)
    def load_video(self):
        video_file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "MP4 Files (*.mp4);;All Files (*)")
        if video_file:
            try:
                if self.cap: self.cap.release()
                self.cap = cv2.VideoCapture(video_file)
                if not self.cap.isOpened(): raise ValueError("Could not open video file.")
                self.video_path = video_file
                self.video_path_label.setText(os.path.basename(video_file))
                num_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if self.keypoints is not None:
                    if num_video_frames != self.keypoints.shape[0]:
                        QMessageBox.warning(self, "Frame Count Mismatch", f"Video: {num_video_frames} frames, Keypoints: {self.keypoints.shape[0]} frames. Using keypoint count.")
                    self.total_frames = self.keypoints.shape[0]
                else: self.total_frames = num_video_frames
                self.frame_index = 0
                self.update_widget_states() # Call state update AFTER cap is set
                self.display_video_frame()
                self.show_status_message(f"Video loaded: {os.path.basename(video_file)} ({self.total_frames} frames)", 5000)
            except Exception as e:
                self.display_error_message("Video Load Error", f"Failed to load video: {e}")
                self.cap = None; self.video_path = None; self.video_path_label.setText("None")
                # Recalculate total_frames based only on keypoints if they exist
                self.total_frames = self.keypoints.shape[0] if self.keypoints is not None else 0
                self.update_widget_states() # Update state after failure


    def load_keypoints(self):
        # print("DEBUG: Entering load_keypoints") # Keep if needed
        file_filter = "Keypoints Files (*.npy *.csv);;All Files (*)"
        keypoints_file, _ = QFileDialog.getOpenFileName(self, "Select Keypoints File", "", file_filter)
        if keypoints_file:
            keypoints_data = load_keypoint_data(keypoints_file)

            if keypoints_data is None:
                self.display_error_message("Keypoint Load Error", f"Failed to load or parse keypoints from:\n{keypoints_file}")
                # print("DEBUG: Exiting load_keypoints (load failed)")
                return

            print(f"Loaded keypoints data type: {keypoints_data.dtype}")

            if keypoints_data.ndim != 3 or keypoints_data.shape[2] != 3:
                 self.display_error_message("Keypoint Data Error", f"Invalid keypoints shape: {keypoints_data.shape}. Expected (frames, points, 3).")
                 # print("DEBUG: Exiting load_keypoints (invalid shape)")
                 return

            # --- Restore Y-Flip ---
            # print("DEBUG: Flipping Y coordinate...")
            keypoints_data[:, :, 1] *= -1
            self.show_status_message("Note: Flipped Y-coordinate of keypoints for OpenGL display.", 4000)
            # print("DEBUG: Y coordinate flipped.")

            self.keypoints = keypoints_data
            self.keypoints_path = keypoints_file
            self.keypoints_path_label.setText(os.path.basename(keypoints_file))
            # print("DEBUG: Assigned keypoints and path.")

            num_keypoint_frames = self.keypoints.shape[0]
            # print(f"DEBUG: num_keypoint_frames = {num_keypoint_frames}")

            # --- Video frame count check ---
            # print("DEBUG: Checking against video frames...")
            if self.cap is not None and self.cap.isOpened():
                num_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # print(f"DEBUG: num_video_frames = {num_video_frames}")
                if num_keypoint_frames != num_video_frames:
                     QMessageBox.warning(self, "Frame Count Mismatch",
                                         f"Keypoints have {num_keypoint_frames} frames, but video has {num_video_frames} frames. "
                                         "Playback/data might be misaligned. Using keypoint count.")
                self.total_frames = num_keypoint_frames # Prioritize keypoint length
            else:
                self.total_frames = num_keypoint_frames
            # print(f"DEBUG: self.total_frames = {self.total_frames}")

            self.frame_index = 0 # Reset frame index
            # print(f"DEBUG: self.frame_index = {self.frame_index}")

            # --- OpenGL data setting still commented out ---
            # print("DEBUG: Skipping set_data")
            self.openGLWidget.set_data(self.keypoints, self.limbSeq)

            # --- Restore subsequent updates ---
            # print("DEBUG: Restoring update_widget_states, display_video_frame, show_status_message")
            self.update_widget_states()
            self.display_video_frame() # Update video frame as well
            self.show_status_message(f"Keypoints loaded: {os.path.basename(keypoints_file)} ({self.total_frames} frames, {self.keypoints.shape[1]} points)", 5000)

            # --- Final print before exiting ---
            # print("DEBUG: Exiting load_keypoints (updates restored)")


    # (Rest of the methods: load_label_names, slider_update_frame, update_frame_display, display_video_frame, label management, saving, closeEvent... remain the same as the previous full version)
    # ... Copy the rest of the methods from the previous full code block here ...
    # --- Frame Navigation & Display ---

    def slider_update_frame(self, value):
        if self.keypoints is not None or self.cap is not None:
            max_idx = self.total_frames - 1 if self.total_frames > 0 else 0
            self.frame_index = max(0, min(value, max_idx))
            self.update_frame_display()

    def update_frame_display(self):
        if not (self.keypoints is not None or self.cap is not None): self.frame_label.setText("Frame: N/A"); return
        max_idx = self.total_frames - 1 if self.total_frames > 0 else 0
        self.frame_index = max(0, min(self.frame_index, max_idx)); self.frame_label.setText(f"Frame: {self.frame_index} / {max_idx}")
        self.slider.blockSignals(True); self.slider.setValue(self.frame_index); self.slider.blockSignals(False)
        self.display_video_frame()
        if self.keypoints is not None and self.openGLWidget.isEnabled(): self.openGLWidget.set_frame_index(self.frame_index)

    def display_video_frame(self):
        if self.cap and self.cap.isOpened():
            video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not (0 <= self.frame_index < video_frames): self.video_label.setText(f"Frame {self.frame_index} out video (0-{video_frames-1})"); return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index); ret, frame = self.cap.read()
            if ret:
                try:
                    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); h,w,ch=frame_rgb.shape; bytes_per_line=ch*w
                    qt_image=QImage(frame_rgb.data,w,h,bytes_per_line,QImage.Format.Format_RGB888)
                    pixmap=QPixmap.fromImage(qt_image); scaled_pixmap=pixmap.scaled(self.video_label.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)
                    self.video_label.setPixmap(scaled_pixmap)
                except Exception as e: print(f"Error display frame {self.frame_index}: {e}"); self.video_label.setText("Error Displaying Frame")
            else: self.video_label.setText(f"Read Error (Frame {self.frame_index})")
        else: self.video_label.setText("No Video Loaded")

    # --- Label Management ---
    def load_label_names(self):
        file_filter = "Text Files (*.txt);;All Files (*)"
        label_file, _ = QFileDialog.getOpenFileName(self, "Select Label Names File (comma-separated)", "", file_filter)
        if label_file:
            try:
                with open(label_file, 'r') as file:
                    content = file.read();
                    if content.startswith('\ufeff'): content = content[1:]
                    label_names = [name.strip() for name in content.split(',') if name.strip()]
                if not label_names: self.show_status_message("Warning: Label names file was empty.", 5000); return
                existing_names={ld.name for ld in self.label_data_list}; new_labels_added=0
                for name in label_names:
                    if name not in existing_names: self.add_label_type(name=name); existing_names.add(name); new_labels_added +=1
                self.show_status_message(f"Loaded {len(label_names)} names. Added {new_labels_added} new types.", 5000); self.update_widget_states()
            except FileNotFoundError: self.display_error_message("Load Error", f"File not found:\n{label_file}")
            except Exception as e: self.display_error_message("Load Error", f"Failed to load names:\n{e}")

    def add_new_label_type(self): self.add_label_type()
    def add_label_type(self, name="", is_numeric=False, values=None):
        if values is None: values = []
        label_data = LabelData(name=name, is_numeric=is_numeric); label_data.values = values
        self.label_data_list.append(label_data); label_widgets = self._create_label_ui(label_data); self.ui_widgets[label_data] = label_widgets; self.update_widget_states()
    def _create_label_ui(self, label_data):
        label_type_widget = QWidget(self); label_type_widget.setStyleSheet("QWidget { border: 1px solid #e0e0e0; margin-bottom: 5px; padding: 3px; }")
        label_type_layout = QVBoxLayout(label_type_widget); label_type_layout.setContentsMargins(2, 2, 2, 2); label_type_layout.setSpacing(3)
        header_layout = QHBoxLayout(); label_name_input = QLineEdit(label_data.name, self); label_name_input.setPlaceholderText("Enter Label Name")
        label_name_input.textChanged.connect(lambda text, ld=label_data: setattr(ld, 'name', text.strip()))
        numeric_checkbox = QCheckBox("Numeric", self); numeric_checkbox.setChecked(label_data.is_numeric)
        numeric_checkbox.stateChanged.connect(lambda state, ld=label_data: setattr(ld, 'is_numeric', state == Qt.CheckState.Checked.value))
        type_collapse_button = QToolButton(self); type_collapse_button.setArrowType(Qt.ArrowType.DownArrow); type_collapse_button.setCheckable(True); type_collapse_button.setChecked(True); type_collapse_button.setStyleSheet("QToolButton { border: none; }")
        header_layout.addWidget(QLabel("Name:")); header_layout.addWidget(label_name_input, stretch=1); header_layout.addWidget(numeric_checkbox); header_layout.addWidget(type_collapse_button)
        label_type_layout.addLayout(header_layout)
        content_area = QWidget(self); content_layout = QVBoxLayout(content_area); content_layout.setContentsMargins(5, 0, 0, 0); content_layout.setSpacing(2)
        label_type_layout.addWidget(content_area); add_value_button = QPushButton("Add Value/Category", self); content_layout.addWidget(add_value_button)
        values_container_layout = QVBoxLayout(); values_container_layout.setSpacing(2); content_layout.addLayout(values_container_layout)
        type_collapse_button.toggled.connect(lambda checked, ca=content_area, btn=type_collapse_button: self._toggle_visibility(checked, ca, btn))
        add_value_button.clicked.connect(lambda _, ld=label_data, vcl=values_container_layout: self._add_value_category(ld, vcl))
        for value_str, intervals_list in label_data.values: self._create_value_ui(label_data, value_str, intervals_list, values_container_layout)
        self.labels_layout.addWidget(label_type_widget)
        return {"main_widget": label_type_widget, "name_input": label_name_input, "numeric_checkbox": numeric_checkbox, "collapse_button": type_collapse_button, "content_area": content_area, "values_container": values_container_layout, "add_value_button": add_value_button}
    def _add_value_category(self, label_data, values_container_layout): new_value = ""; new_intervals = []; label_data.values.append((new_value, new_intervals)); self._create_value_ui(label_data, new_value, new_intervals, values_container_layout)
    def _find_value_data_index(self, label_data, value_str, intervals_list_ref):
         for i, (v_str, i_list) in enumerate(label_data.values):
             if i_list is intervals_list_ref: return i
         print(f"Warning: Find value index fail '{value_str}'. Trying by value."); # Shortened warning
         for i, (v_str, i_list) in enumerate(label_data.values):
             if v_str == value_str: return i
         return -1
    def _create_value_ui(self, label_data, value_str, intervals_list_ref, values_container_layout):
        value_data_index = self._find_value_data_index(label_data, value_str, intervals_list_ref)
        if value_data_index == -1: print(f"Error: Find data tuple fail '{value_str}'."); return # Shortened error
        value_widget = QWidget(self); value_widget.setStyleSheet("QWidget { border: 1px solid #f5f5f5; margin-bottom: 3px; padding: 2px; }")
        value_layout = QVBoxLayout(value_widget); value_layout.setContentsMargins(2, 2, 2, 2); value_layout.setSpacing(2)
        value_header_layout = QHBoxLayout(); value_input = QLineEdit(value_str, self); value_input.setPlaceholderText("Enter Value/Category")
        value_input.textChanged.connect(lambda text, ld=label_data, idx=value_data_index: self._update_value_string(ld, idx, text))
        value_collapse_button = QToolButton(self); value_collapse_button.setArrowType(Qt.ArrowType.DownArrow); value_collapse_button.setCheckable(True); value_collapse_button.setChecked(True); value_collapse_button.setStyleSheet("QToolButton { border: none; }")
        delete_value_button = QPushButton("X", self); delete_value_button.setFixedSize(QSize(20, 20)); delete_value_button.setStyleSheet("QPushButton { color: red; font-weight: bold; border: none; }"); delete_value_button.setToolTip("Delete this value/category")
        value_header_layout.addWidget(QLabel("Value:")); value_header_layout.addWidget(value_input, stretch=1); value_header_layout.addWidget(value_collapse_button); value_header_layout.addWidget(delete_value_button); value_layout.addLayout(value_header_layout)
        intervals_area = QWidget(self); intervals_area_layout = QVBoxLayout(intervals_area); intervals_area_layout.setContentsMargins(5, 0, 0, 0); intervals_area_layout.setSpacing(2); value_layout.addWidget(intervals_area)
        add_interval_button = QPushButton("Add Interval", self); intervals_area_layout.addWidget(add_interval_button); intervals_container_layout = QVBoxLayout(); intervals_area_layout.addLayout(intervals_container_layout)
        value_collapse_button.toggled.connect(lambda checked, ia=intervals_area, btn=value_collapse_button: self._toggle_visibility(checked, ia, btn))
        add_interval_button.clicked.connect(lambda _, ld=label_data, il_ref=intervals_list_ref, icl=intervals_container_layout: self._add_interval(ld, il_ref, icl))
        delete_value_button.clicked.connect(lambda _, ld=label_data, idx=value_data_index, vw=value_widget: self._delete_value_category(ld, idx, vw))
        for interval_pair_ref in intervals_list_ref:
             if isinstance(interval_pair_ref, list) and len(interval_pair_ref) == 2: self._create_interval_ui(label_data, intervals_list_ref, interval_pair_ref, intervals_container_layout)
             else: print(f"Warn: Skip malformed interval {interval_pair_ref} for '{value_str}'") # Shortened warn
        values_container_layout.addWidget(value_widget)
    def _update_value_string(self, label_data, index, new_text):
         if 0 <= index < len(label_data.values): label_data.values[index] = (new_text.strip(), label_data.values[index][1])
         else: print(f"Error update value string: index {index} out bounds.") # Shortened error
    def _delete_value_category(self, label_data, index, value_widget):
         if 0 <= index < len(label_data.values): label_data.values.pop(index); value_widget.deleteLater(); self.show_status_message("Value category removed.", 3000) # Longer message time
         else: print(f"Error deleting value: index {index} out bounds.") # Shortened error
    def _add_interval(self, label_data, intervals_list_ref, intervals_container_layout): new_interval_pair = [0, 0]; intervals_list_ref.append(new_interval_pair); self._create_interval_ui(label_data, intervals_list_ref, new_interval_pair, intervals_container_layout)
    def _create_interval_ui(self, label_data, intervals_list_ref, interval_pair_ref, intervals_container_layout):
        start_frame = interval_pair_ref[0] if len(interval_pair_ref) > 0 else 0; end_frame = interval_pair_ref[1] if len(interval_pair_ref) > 1 else 0
        interval_widget = QWidget(self); interval_layout = QHBoxLayout(interval_widget); interval_layout.setContentsMargins(0, 0, 0, 0)
        start_frame_input = QLineEdit(str(start_frame), self); end_frame_input = QLineEdit(str(end_frame), self)
        start_frame_input.textChanged.connect(lambda text, ref=interval_pair_ref: self._update_interval_frame(ref, 0, text))
        end_frame_input.textChanged.connect(lambda text, ref=interval_pair_ref: self._update_interval_frame(ref, 1, text))
        delete_interval_button = QPushButton("x", self); delete_interval_button.setFixedSize(QSize(15, 15)); delete_interval_button.setStyleSheet("QPushButton { color: red; border: none; }"); delete_interval_button.setToolTip("Delete interval")
        interval_layout.addWidget(QLabel("Start:")); interval_layout.addWidget(start_frame_input); interval_layout.addWidget(QLabel("End:")); interval_layout.addWidget(end_frame_input); interval_layout.addWidget(delete_interval_button)
        delete_interval_button.clicked.connect(lambda _, il_ref=intervals_list_ref, ip_ref=interval_pair_ref, iw=interval_widget: self._delete_interval(il_ref, ip_ref, iw))
        intervals_container_layout.addWidget(interval_widget)
    def _update_interval_frame(self, interval_pair_ref, index_in_pair, text):
         try:
             if isinstance(interval_pair_ref, list) and 0 <= index_in_pair < 2: interval_pair_ref[index_in_pair] = int(text.strip()) # Check length before access
             else: print(f"Error update interval: Invalid ref/index."); interval_pair_ref[index_in_pair] = 0 # Simplified error handling
         except ValueError: interval_pair_ref[index_in_pair] = 0; self.show_status_message(f"Warn: Invalid frame input '{text}'. Reset 0.", 3000) # Shortened msg
         except Exception as e: print(f"Error update interval frame: {e}")
    def _delete_interval(self, intervals_list_ref, interval_pair_ref, interval_widget):
        try: intervals_list_ref.remove(interval_pair_ref); interval_widget.deleteLater(); self.show_status_message("Interval removed.", 2000)
        except ValueError: print(f"Error: Interval {interval_pair_ref} not found.") # Shortened error
        except Exception as e: print(f"Error deleting interval widget: {e}")
    def _toggle_visibility(self, checked, widget_to_toggle, button): widget_to_toggle.setVisible(checked); button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    # --- Saving ---
    def save_csv(self):
        # (Method remains the same as previous version)
        if self.keypoints is None: self.display_error_message("Save Error", "No keypoints loaded."); return
        has_named_labels = any(ld.name for ld in self.label_data_list if hasattr(ld, 'name'))
        if not has_named_labels: self.display_error_message("Save Error", "No labels defined/named."); return
        base_name = os.path.splitext(self.keypoints_path)[0] if self.keypoints_path else "classic_labeled"; suggested_path = f"{base_name}_labels.csv"
        save_file, _ = QFileDialog.getSaveFileName(self, "Save CSV", suggested_path, "CSV Files (*.csv)")
        if not save_file: self.show_status_message("Save cancelled.", 2000); return
        climber_id = self.climber_id_input.text().strip(); route_id = self.route_id_input.text().strip()
        self.show_status_message(f"Saving to {save_file}..."); QApplication.processEvents()
        data_to_save = []; total_frames = self.keypoints.shape[0]; warnings_list = set()
        try:
            headers = ["climber_id", "route_id", "frame"]; num_kps = self.keypoints.shape[1]
            for i in range(num_kps): headers.extend([f"kp{i}_x", f"kp{i}_y", f"kp{i}_z"])
            label_names = []; valid_label_data = []
            for ld in self.label_data_list:
                 if ld.name:
                      if ld.name in label_names: raise ValueError(f"Duplicate label name '{ld.name}'.")
                      label_names.append(ld.name); valid_label_data.append(ld)
                 else: warnings.warn("Skip empty label name.", UserWarning); warnings_list.add("empty_label")
            if not valid_label_data: raise ValueError("No valid labels with names.")
            headers.extend(label_names)
            for frame_idx in range(total_frames):
                frame_data = {"climber_id": climber_id, "route_id": route_id, "frame": frame_idx}
                for kp_idx, point in enumerate(self.keypoints[frame_idx]):
                    if len(point)>=3: frame_data[f"kp{kp_idx}_x"]=point[0]; frame_data[f"kp{kp_idx}_y"]=point[1]; frame_data[f"kp{kp_idx}_z"]=point[2]
                    else: frame_data[f"kp{kp_idx}_x"]=np.nan; frame_data[f"kp{kp_idx}_y"]=np.nan; frame_data[f"kp{kp_idx}_z"]=np.nan; warnings_list.add(f"kp_dim_{kp_idx}")
                for label_data in valid_label_data:
                    label_name = label_data.name; value_for_frame = None
                    for value_str, intervals_list in label_data.values:
                        for interval_pair in intervals_list:
                            try:
                                if not (isinstance(interval_pair, list) and len(interval_pair) == 2): raise IndexError
                                start_f=int(interval_pair[0]); end_f=int(interval_pair[1])
                                if start_f <= frame_idx <= end_f: value_for_frame = value_str; break
                            except (ValueError, TypeError, IndexError): warnings_list.add(f"interval_{label_name}"); continue # Simplified warning key
                        if value_for_frame is not None: break
                    if value_for_frame is not None:
                        if label_data.is_numeric:
                            try: frame_data[label_name] = float(value_for_frame)
                            except (ValueError, TypeError): frame_data[label_name] = 0.0; warnings_list.add(f"numeric_{label_name}") # Simplified warning key
                        else: frame_data[label_name] = value_for_frame
                    else: frame_data[label_name] = 0.0 if label_data.is_numeric else ""
                data_to_save.append(frame_data)
                if (frame_idx + 1) % 100 == 0: self.show_status_message(f"Saving... {frame_idx+1}/{total_frames}", 0); QApplication.processEvents()
            df = pd.DataFrame(data_to_save, columns=headers); df.to_csv(save_file, index=False, na_rep='NaN')
            if warnings_list: print(f"\n--- Save Warnings ---\nUnique issues encountered: {', '.join(sorted(list(warnings_list)))}\n-------------------\n")
            self.show_status_message(f"Saved to {save_file}", 5000)
        except ValueError as ve: self.display_error_message("Save Error", f"{ve}"); self.show_status_message("Save failed.", 5000)
        except Exception as e: self.display_error_message("Save Error", f"Unexpected save error:\n{e}"); self.show_status_message("Save failed.", 5000)

    # --- Window Closing ---
    def closeEvent(self, event: QCloseEvent):
        if self.cap: self.cap.release()
        event.accept()