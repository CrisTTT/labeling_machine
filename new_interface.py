import sys
import os
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import (QMainWindow, QHBoxLayout, QVBoxLayout, QLabel,
                             QWidget, QPushButton, QFileDialog, QLineEdit,
                             QFormLayout, QCheckBox, QScrollArea, QMessageBox,
                             QSizePolicy, QStatusBar, QSlider,
                             QApplication)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QCloseEvent

from open_gl_widget import OpenGLWidget
from utils import load_keypoint_data
import warnings


class LabelUIElements:
    def __init__(self, name_widget, value_widget, numeric_checkbox, container_widget):
        self.name_widget = name_widget
        self.value_widget = value_widget
        self.numeric_checkbox = numeric_checkbox
        self.container_widget = container_widget


class NewInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.keypoints = None;
        self.video_path = None;
        self.keypoints_path = None;
        self.cap = None
        self.total_frames = 0;
        self.frame_index = 0
        self.limbSeq = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14],
                        [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
        self.label_names = [];
        self.label_is_numeric = {};
        self.label_ui_elements = {};
        self.label_values = {}
        self.csv_file = None;
        self._has_unsaved_changes = False
        self.initUI();
        self.update_widget_states()

    def initUI(self):
        self.setWindowTitle('New Frame-by-Frame 3D Pose Labeling Tool');
        self.setGeometry(100, 100, 1300, 700)
        self.statusBar = QStatusBar(self);
        self.setStatusBar(self.statusBar);
        self.statusBar.showMessage("Ready. Load video and keypoints to begin.")
        central_widget = QWidget(self);
        self.setCentralWidget(central_widget);
        main_layout = QHBoxLayout(central_widget)
        left_panel = QVBoxLayout()
        self.video_label = QLabel("No Video Loaded", self);
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter);
        self.video_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding);
        self.video_label.setMinimumSize(480, 270);
        left_panel.addWidget(self.video_label, stretch=1)
        self.openGLWidget = OpenGLWidget(self);
        self.openGLWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding);
        self.openGLWidget.setMinimumSize(480, 270)
        self.openGLWidget.statusUpdateRequest.connect(self.show_status_message);
        left_panel.addWidget(self.openGLWidget, stretch=1)
        nav_controls_layout = QVBoxLayout();
        button_nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("< Previous", self);
        self.prev_button.clicked.connect(self.prev_frame);
        button_nav_layout.addWidget(self.prev_button)
        self.frame_label = QLabel(f"Frame: {self.frame_index} / {self.total_frames - 1}", self);
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter);
        button_nav_layout.addWidget(self.frame_label, stretch=1)
        self.next_button = QPushButton("Next >", self);
        self.next_button.clicked.connect(self.next_frame);
        button_nav_layout.addWidget(self.next_button)
        self.copy_last_button = QPushButton("Copy from Last Frame", self);
        self.copy_last_button.setToolTip("Copy all label values from the previous frame to the current frame.")
        self.copy_last_button.clicked.connect(self.copy_labels_from_previous_frame);
        button_nav_layout.addWidget(self.copy_last_button)
        nav_controls_layout.addLayout(button_nav_layout)
        self.slider = QSlider(Qt.Orientation.Horizontal, self);
        self.slider.setMinimum(0);
        self.slider.setMaximum(0);
        self.slider.valueChanged.connect(self.slider_update_frame);
        nav_controls_layout.addWidget(self.slider)
        left_panel.addLayout(nav_controls_layout);
        main_layout.addLayout(left_panel, stretch=2)
        right_panel = QVBoxLayout();
        right_panel.setSpacing(10)
        file_layout = QFormLayout()
        self.load_video_button = QPushButton("Load Video (.mp4)", self);
        self.load_video_button.clicked.connect(self.load_video)
        self.load_keypoints_button = QPushButton("Load Keypoints (.npy, .csv)", self);
        self.load_keypoints_button.clicked.connect(self.load_keypoints)
        self.video_path_label = QLabel("None", self);
        self.keypoints_path_label = QLabel("None", self)
        file_layout.addRow(self.load_video_button, self.video_path_label);
        file_layout.addRow(self.load_keypoints_button, self.keypoints_path_label);
        right_panel.addLayout(file_layout)
        id_layout = QFormLayout()
        self.climber_id_input = QLineEdit("climber_001", self);
        self.route_id_input = QLineEdit("route_001", self)
        id_layout.addRow("Subject ID:", self.climber_id_input);
        id_layout.addRow("Action ID:", self.route_id_input)
        right_panel.addLayout(id_layout)
        label_controls_layout = QHBoxLayout()
        self.load_labels_button = QPushButton("Load Label Names (.txt)", self);
        self.load_labels_button.clicked.connect(self.load_label_names_and_init_data)
        self.save_button = QPushButton("Save Labels to CSV", self);
        self.save_button.clicked.connect(self.save_csv)
        label_controls_layout.addWidget(self.load_labels_button);
        label_controls_layout.addWidget(self.save_button);
        right_panel.addLayout(label_controls_layout)
        right_panel.addWidget(QLabel("--- Labels (Current Frame Values) ---"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.labels_scroll_area = QScrollArea(self);
        self.labels_scroll_area.setWidgetResizable(True);
        self.labels_scroll_area.setStyleSheet("background-color: #ffffff;")
        self.labels_widget = QWidget();
        self.labels_layout = QVBoxLayout(self.labels_widget);
        self.labels_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.labels_scroll_area.setWidget(self.labels_widget);
        right_panel.addWidget(self.labels_scroll_area, stretch=1);
        main_layout.addLayout(right_panel, stretch=1)
        self.show()

    def update_widget_states(self):
        has_video = self.cap is not None and self.cap.isOpened();
        has_keypoints = self.keypoints is not None;
        has_labels = bool(self.label_names)
        can_copy = has_keypoints and has_labels and self.frame_index > 0;
        data_loaded = has_keypoints or has_video
        self.prev_button.setEnabled(data_loaded);
        self.next_button.setEnabled(data_loaded);
        self.slider.setEnabled(data_loaded)
        self.openGLWidget.setEnabled(has_keypoints);
        self.save_button.setEnabled(has_keypoints and has_labels and self.csv_file is not None);
        self.copy_last_button.setEnabled(can_copy)
        for elements in self.label_ui_elements.values(): elements.value_widget.setEnabled(has_keypoints)
        if data_loaded:
            self.slider.setMaximum(self.total_frames - 1 if self.total_frames > 0 else 0)
        else:
            self.slider.setMaximum(0); self.frame_index = 0; self.frame_label.setText("Frame: N/A")
        if data_loaded: self.update_frame_display()

    def show_status_message(self, message, timeout=3000):
        self.statusBar.showMessage(message, timeout)

    def display_error_message(self, title, message):
        QMessageBox.critical(self, title, message); self.show_status_message(f"Error: {message}", 5000)

    def load_video(self):
        video_file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "MP4 Files (*.mp4);;All Files (*)")
        if video_file:
            try:
                if self.cap: self.cap.release()
                self.cap = cv2.VideoCapture(video_file);
                if not self.cap.isOpened(): raise ValueError("Could not open video file.")
                self.video_path = video_file;
                self.video_path_label.setText(os.path.basename(video_file))
                num_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if self.keypoints is not None:
                    if num_video_frames != self.keypoints.shape[0]: QMessageBox.warning(self, "Frame Count Mismatch",
                                                                                        f"Video: {num_video_frames}, Keypoints: {self.keypoints.shape[0]}. Using keypoint count.")
                    self.total_frames = self.keypoints.shape[0]
                else:
                    self.total_frames = num_video_frames
                self.frame_index = 0;
                self.update_widget_states()
                self.show_status_message(f"Video loaded: {os.path.basename(video_file)} ({self.total_frames} frames)",
                                         5000)
            except Exception as e:
                self.display_error_message("Video Load Error", f"{e}");
                self.cap = None;
                self.video_path = None;
                self.video_path_label.setText("None")
                self.total_frames = self.keypoints.shape[0] if self.keypoints is not None else 0;
                self.update_widget_states()

    def load_keypoints(self):
        file_filter = "Keypoints Files (*.npy *.csv);;All Files (*)";
        keypoints_file, _ = QFileDialog.getOpenFileName(self, "Select Keypoints File", "", file_filter)
        if keypoints_file:
            keypoints_data = load_keypoint_data(keypoints_file)  # Uses updated utils.py
            if keypoints_data is None: self.display_error_message("Keypoint Load Error",
                                                                  f"Failed load: {keypoints_file}"); return
            if keypoints_data.ndim != 3 or keypoints_data.shape[-1] != 3: self.display_error_message(
                "Keypoint Data Error", f"Invalid shape: {keypoints_data.shape}. Expected (frames, points, 3)."); return

            if keypoints_file.endswith('.npy'):
                keypoints_data[:, :, 1] *= -1
                keypoints_data[:, :, 0] *= -1
            elif keypoints_file.endswith('.csv'):
                keypoints_data[:, :, 0] *= -1

            self.keypoints = keypoints_data;
            self.keypoints_path = keypoints_file;
            self.keypoints_path_label.setText(os.path.basename(keypoints_file))
            num_keypoint_frames = self.keypoints.shape[0]
            if self.cap is not None and self.cap.isOpened():
                num_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if num_keypoint_frames != num_video_frames: QMessageBox.warning(self, "Frame Count Mismatch",
                                                                                f"Keypoints: {num_keypoint_frames}, Video: {num_video_frames}. Using keypoint count.")
                self.total_frames = num_keypoint_frames
            else:
                self.total_frames = num_keypoint_frames
            self.frame_index = 0;
            self.openGLWidget.set_data(self.keypoints, self.limbSeq)
            if self.label_names: self._load_or_initialize_label_data()
            self.update_widget_states()
            self.show_status_message(
                f"Keypoints loaded: {os.path.basename(keypoints_file)} ({self.total_frames} frames, {self.keypoints.shape[1]} points)",
                5000)

    def load_label_names_and_init_data(self):
        if self.keypoints is None: QMessageBox.warning(self, "Load Order", "Load keypoints before labels."); return
        file_filter = "Text Files (*.txt);;All Files (*)";
        label_file, _ = QFileDialog.getOpenFileName(self, "Select Label Names File", "", file_filter)
        if not label_file: return
        try:
            with open(label_file, 'r') as file:
                content = file.read();
            if content.startswith('\ufeff'): content = content[1:]
            loaded_names = [name.strip() for name in content.split(',') if name.strip()]
            if not loaded_names: self.show_status_message("Warning: Label names file empty.", 5000); return
            self._clear_labels();
            self.label_names = loaded_names
            for name in self.label_names:
                self.label_is_numeric[name] = False
                self._add_label_ui(name)
            self.show_status_message(f"Loaded {len(self.label_names)} names. Initializing...", 0);
            QApplication.processEvents()
            self._load_or_initialize_label_data()
        except FileNotFoundError:
            self.display_error_message("Load Error", f"File not found: {label_file}")
        except Exception as e:
            self.display_error_message("Load Error",
                                       f"Failed load names: {e}"); self._clear_labels(); self.update_widget_states()

    def _clear_labels(self):
        self.label_names = [];
        self.label_is_numeric = {};
        self.label_values = {};
        self.csv_file = None;
        self._has_unsaved_changes = False
        for label_name in list(self.label_ui_elements.keys()):
            elements = self.label_ui_elements.pop(label_name)
            if elements.container_widget: elements.container_widget.deleteLater()
        while self.labels_layout.count():
            child = self.labels_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

    def _add_label_ui(self, label_name):
        if label_name in self.label_ui_elements: print(f"Warn: UI exists '{label_name}'."); return
        label_container_widget = QWidget(self);
        row_layout = QHBoxLayout(label_container_widget);
        row_layout.setContentsMargins(0, 0, 0, 0)
        label_name_display = QLabel(label_name);
        label_value_input = QLineEdit(self);
        label_value_input.setEnabled(self.keypoints is not None)
        numeric_label_checkbox = QCheckBox("Numeric (Zero Padding)", self);
        delete_button = QPushButton("X", self);
        delete_button.setFixedSize(QSize(22, 22))
        delete_button.setToolTip(f"Delete label '{label_name}'");
        delete_button.setStyleSheet("QPushButton { color: red; font-weight: bold; }")
        delete_button.clicked.connect(lambda checked=False, name=label_name: self._prompt_delete_label(name))
        row_layout.addWidget(label_name_display);
        row_layout.addWidget(label_value_input, 1);
        row_layout.addWidget(numeric_label_checkbox);
        row_layout.addWidget(delete_button)
        self.labels_layout.addWidget(label_container_widget)
        ui_elements = LabelUIElements(label_name_display, label_value_input, numeric_label_checkbox,
                                      label_container_widget)
        self.label_ui_elements[label_name] = ui_elements
        numeric_label_checkbox.setChecked(self.label_is_numeric.get(label_name, False))
        numeric_label_checkbox.stateChanged.connect(
            lambda state, name=label_name: self.label_is_numeric.update({name: state == Qt.CheckState.Checked.value}))
        label_value_input.textChanged.connect(lambda text, name=label_name, iw=label_value_input,
                                                     cb=numeric_label_checkbox: self._update_label_data_from_input(name,
                                                                                                                   text,
                                                                                                                   iw,
                                                                                                                   cb))
        value = self.label_values.get(self.frame_index, {}).get(label_name, "");
        label_value_input.setText(str(value))

    def _prompt_delete_label(self, label_name_to_delete):
        if label_name_to_delete not in self.label_names: self.show_status_message(
            f"Label '{label_name_to_delete}' not found.", 3000); return
        reply = QMessageBox.question(self, 'Confirm Delete', f"Delete label '{label_name_to_delete}' and all its data?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes: self._delete_label_data(label_name_to_delete)

    def _delete_label_data(self, label_name_to_delete):
        try:
            if label_name_to_delete in self.label_ui_elements:
                elements_to_remove = self.label_ui_elements.pop(label_name_to_delete)
                if elements_to_remove.container_widget: elements_to_remove.container_widget.deleteLater()
            if label_name_to_delete in self.label_names: self.label_names.remove(label_name_to_delete)
            if label_name_to_delete in self.label_is_numeric: del self.label_is_numeric[label_name_to_delete]
            for frame_idx in self.label_values:
                if label_name_to_delete in self.label_values[frame_idx]: del self.label_values[frame_idx][
                    label_name_to_delete]
            self._has_unsaved_changes = True;
            self.update_widget_states()
            self.show_status_message(f"Label '{label_name_to_delete}' deleted.", 3000)
        except Exception as e:
            self.display_error_message("Delete Error", f"Could not delete '{label_name_to_delete}': {e}")

    def _load_or_initialize_label_data(self):
        if self.keypoints is None or not self.label_names:
            self.show_status_message("Cannot initialize: Load keypoints and label names first.", 5000)
            return
        if not self.keypoints_path:
            self.display_error_message("Data Error", "Keypoints path is missing for CSV determination.")
            return

        base_name = os.path.splitext(self.keypoints_path)[0]
        self.csv_file = f"{base_name}_newlabels.csv"
        self.label_values = {}
        self._has_unsaved_changes = False

        try:
            if os.path.exists(self.csv_file):
                self.show_status_message(f"Loading existing labels from {self.csv_file}...", 0)
                # Read all as strings, na_filter=False means empty strings "" are read as "", not NaN by pandas.
                df = pd.read_csv(self.csv_file, dtype=str, na_filter=False)

                if "frame" not in df.columns:
                    raise ValueError("Existing CSV is missing the required 'frame' column.")

                # df["frame"] is already string due to dtype=str.

                for frame_idx_str in df["frame"].unique():
                    frame = int(frame_idx_str)
                    if frame not in self.label_values:
                        self.label_values[frame] = {}

                    row_series = df[df["frame"] == frame_idx_str].iloc[0]

                    for label_name in self.label_names:
                        is_numeric = self.label_is_numeric.get(label_name, False)
                        default_value = 0.0 if is_numeric else ""

                        value_from_csv_str = ""  # Default if column not found or cell is empty
                        if label_name in df.columns:
                            raw_cell_value = row_series.get(label_name)  # Will be a string
                            value_from_csv_str = str(raw_cell_value)  # Ensure it's a string

                        if value_from_csv_str.strip() == "":
                            value = default_value
                        else:  # We have a non-empty string from the CSV
                            if is_numeric:
                                try:
                                    value = float(value_from_csv_str)
                                except ValueError:
                                    # This case is for when a non-empty string cannot be converted to float
                                    # e.g. "text" in a numeric column. Test case: test_load_csv_malformed_numeric_data
                                    value = 0.0
                            else:
                                value = value_from_csv_str  # Use the non-empty string as is
                        self.label_values[frame][label_name] = value
                self.show_status_message(f"Successfully loaded labels from {self.csv_file}", 5000)

            else:
                self.show_status_message(f"No existing label file found. Creating new file: {self.csv_file}", 0)
                num_frames = self.keypoints.shape[0]
                for frame_idx in range(num_frames):
                    self.label_values[frame_idx] = {
                        name: (0.0 if self.label_is_numeric.get(name, False) else "")
                        for name in self.label_names
                    }
                self.show_status_message(f"Initialized empty labels for {num_frames} frames. Save to create CSV.", 5000)

        except FileNotFoundError:
            self.display_error_message("Load Error", f"Could not find CSV file: {self.csv_file}")
            self.csv_file = None
        except (ValueError, KeyError, pd.errors.ParserError, Exception) as e:
            self.display_error_message("Load Error",
                                       f"Failed to load or parse existing CSV ({self.csv_file}):\n{e}\n\nRe-initializing with empty labels.")
            self.label_values = {}
            self.csv_file = f"{base_name}_newlabels.csv"
            num_frames = self.keypoints.shape[0]
            for frame_idx in range(num_frames):
                self.label_values[frame_idx] = {name: (0.0 if self.label_is_numeric.get(name, False) else "") for name
                                                in self.label_names}

        self.update_label_value_inputs()
        self.update_widget_states()

    def prev_frame(self):
        if self.total_frames > 0 and self.frame_index > 0: self.frame_index -= 1; self.update_frame_display()

    def next_frame(self):
        if self.total_frames > 0 and self.frame_index < self.total_frames - 1: self.frame_index += 1; self.update_frame_display()

    def slider_update_frame(self, value):
        if self.keypoints is not None or self.cap is not None:
            max_idx = self.total_frames - 1 if self.total_frames > 0 else 0
            self.frame_index = max(0, min(value, max_idx));
            self.update_frame_display()

    def update_frame_display(self):
        if not (self.keypoints is not None or self.cap is not None):
            self.frame_label.setText("Frame: N/A");
            self.copy_last_button.setEnabled(False)
            if hasattr(self, 'slider'): self.slider.setEnabled(False); return
        max_idx = self.total_frames - 1 if self.total_frames > 0 else 0
        self.frame_index = max(0, min(self.frame_index, max_idx))
        self.frame_label.setText(f"Frame: {self.frame_index} / {max_idx}")
        if hasattr(self, 'slider'): self.slider.blockSignals(True); self.slider.setValue(
            self.frame_index); self.slider.blockSignals(False)
        self.display_video_frame()
        if self.keypoints is not None:
            self.openGLWidget.set_frame_index(self.frame_index)
        else:
            self.openGLWidget.update()
        self.update_label_value_inputs()
        self.copy_last_button.setEnabled(self.keypoints is not None and bool(self.label_names) and self.frame_index > 0)

    def display_video_frame(self):
        if self.cap and self.cap.isOpened():
            video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not (0 <= self.frame_index < video_frames): self.video_label.setText(
                f"Frame {self.frame_index} out video (0-{video_frames - 1})"); return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index);
            ret, frame = self.cap.read()
            if ret:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);
                    h, w, ch = frame_rgb.shape;
                    bytes_per_line = ch * w
                    qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image);
                    scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)
                    self.video_label.setPixmap(scaled_pixmap)
                except Exception as e:
                    print(f"Error display frame {self.frame_index}: {e}");
                    self.video_label.setText(
                        "Error Displaying Frame")
            else:
                self.video_label.setText(f"Read Error (Frame {self.frame_index})")
        else:
            self.video_label.setText("No Video Loaded")

    def update_label_value_inputs(self):
        current_frame_values = self.label_values.get(self.frame_index, {})
        for label_name, elements in self.label_ui_elements.items():
            value = current_frame_values.get(label_name, "")
            elements.value_widget.blockSignals(True);
            elements.value_widget.setText(str(value));
            elements.value_widget.blockSignals(False)
            elements.numeric_checkbox.blockSignals(True);
            elements.numeric_checkbox.setChecked(self.label_is_numeric.get(label_name, False));
            elements.numeric_checkbox.blockSignals(False)

    def _update_label_data_from_input(self, label_name, text, input_widget, numeric_checkbox):
        if self.frame_index not in self.label_values: self.label_values[self.frame_index] = {}
        is_numeric = numeric_checkbox.isChecked();
        current_value = None;
        parse_error = False
        if is_numeric:
            try:
                current_value = float(text.strip())
            except ValueError:
                input_widget.setStyleSheet("QLineEdit { background-color: #ffdddd; }");
                self.show_status_message(
                    f"Warn: Invalid numeric '{text}' for {label_name}.", 3000);
                parse_error = True;
                current_value = 0.0
            else:
                input_widget.setStyleSheet("")
        else:
            current_value = text;
            input_widget.setStyleSheet("")

        previous_value = self.label_values[self.frame_index].get(label_name)

        if (is_numeric and parse_error) or \
                (not parse_error and previous_value != current_value):
            self.label_values[self.frame_index][label_name] = current_value;
            self._has_unsaved_changes = True

    def copy_labels_from_previous_frame(self):
        if self.frame_index <= 0: self.show_status_message("Cannot copy: First frame.", 3000); return
        if not self.label_values: self.show_status_message("Cannot copy: No label data.", 3000); return
        prev_frame_index = self.frame_index - 1
        prev_frame_data = self.label_values.get(prev_frame_index, None)
        if prev_frame_data is None: self.show_status_message(f"Cannot copy: No data for frame {prev_frame_index}.",
                                                             3000); return
        if self.frame_index not in self.label_values: self.label_values[self.frame_index] = {}
        self.label_values[self.frame_index] = prev_frame_data.copy()
        self._has_unsaved_changes = True;
        self.update_label_value_inputs()
        self.show_status_message(f"Copied labels from frame {prev_frame_index}.", 2000)

    def save_csv(self, show_dialog=True):
        if self.keypoints is None: self.display_error_message("Save Error", "No keypoints loaded."); return
        if not self.label_names: self.display_error_message("Save Error", "No labels defined."); return
        save_path = self.csv_file
        if not self.csv_file or show_dialog:
            suggested_path = self.csv_file if self.csv_file else f"{os.path.splitext(self.keypoints_path)[0]}_newlabels.csv" if self.keypoints_path else "keypoints_newlabels.csv"
            save_path_selected, _ = QFileDialog.getSaveFileName(self, "Save Labels CSV", suggested_path,
                                                                "CSV Files (*.csv)")
            if not save_path_selected: self.show_status_message("Save cancelled.", 2000); return
            save_path = save_path_selected;
            self.csv_file = save_path
        climber_id = self.climber_id_input.text().strip();
        route_id = self.route_id_input.text().strip()
        self.show_status_message(f"Saving to {save_path}...");
        QApplication.processEvents()
        data_to_save = [];
        num_frames = self.keypoints.shape[0]
        try:
            headers = ["climber_id", "route_id", "frame"];
            num_kps = self.keypoints.shape[1]
            for i in range(num_kps): headers.extend([f"kp{i}_x", f"kp{i}_y", f"kp{i}_z"])
            headers.extend(self.label_names)
            for frame_idx in range(num_frames):
                frame_data = {"climber_id": climber_id, "route_id": route_id, "frame": frame_idx}
                for kp_idx, point in enumerate(self.keypoints[frame_idx]):
                    frame_data[f"kp{kp_idx}_x"] = point[0];
                    frame_data[f"kp{kp_idx}_y"] = point[1];
                    frame_data[f"kp{kp_idx}_z"] = point[2]
                frame_labels = self.label_values.get(frame_idx, {})
                for label_name in self.label_names:
                    is_numeric = self.label_is_numeric.get(label_name, False);
                    default_value = 0.0 if is_numeric else ""
                    value = frame_labels.get(label_name,
                                             default_value)
                    if is_numeric:
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            value = 0.0
                    else:
                        value = str(value)
                    frame_data[label_name] = value
                data_to_save.append(frame_data)
                if (frame_idx + 1) % 100 == 0: self.show_status_message(f"Saving... {frame_idx + 1}/{num_frames}",
                                                                        0); QApplication.processEvents()
            df = pd.DataFrame(data_to_save, columns=headers);
            df.to_csv(save_path, index=False);
            self._has_unsaved_changes = False
            self.show_status_message(f"Saved to {save_path}", 5000)
        except Exception as e:
            self.display_error_message("Save Error", f"Unexpected save error:\n{e}");
            self.show_status_message(
                "Save failed.", 5000)

    def closeEvent(self, event: QCloseEvent):
        if self._has_unsaved_changes:
            reply = QMessageBox.question(self, 'Unsaved Changes', "Save before closing?",
                                         QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                                         QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Save: self.save_csv(show_dialog=False);
            if self._has_unsaved_changes:
                event.ignore();
                return
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore();
                return
        if self.cap: self.cap.release()
        event.accept()
