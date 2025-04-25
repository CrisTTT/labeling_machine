# new_interface.py
import sys
import os
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
from PyQt6.QtWidgets import (QMainWindow, QHBoxLayout, QVBoxLayout, QLabel,
                             QWidget, QPushButton, QFileDialog, QLineEdit,
                             QFormLayout, QCheckBox, QScrollArea, QMessageBox,
                             QSizePolicy, QStatusBar)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QCloseEvent
from PyQt6.QtWidgets import QApplication
# Assuming revised OpenGLWidget with set_data, set_frame_index, statusUpdateRequest
from open_gl_widget import OpenGLWidget
from utils import load_keypoint_data # Assuming revised utils
import warnings


# Helper structure to link UI elements for a label type
class LabelUIElements:
     def __init__(self, name_widget, value_widget, numeric_checkbox):
         self.name_widget = name_widget
         self.value_widget = value_widget
         self.numeric_checkbox = numeric_checkbox

class NewInterface(QMainWindow):
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

        # Label related data
        self.label_names = [] # List of strings
        self.label_is_numeric = {} # Map: label_name -> bool
        self.label_ui_elements = {} # Map: label_name -> LabelUIElements
        self.label_values = {} # Nested Dict: {frame_idx: {label_name: value}}
        self.csv_file = None # Path to the currently associated CSV file

        # Flag for unsaved changes (optional - for close confirmation)
        self._has_unsaved_changes = False

        self.initUI()
        self.update_widget_states() # Initial state

    def initUI(self):
        self.setWindowTitle('New Frame-by-Frame 3D Pose Labeling Tool')
        self.setGeometry(100, 100, 1300, 700)

        # --- Status Bar ---
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready. Load video and keypoints to begin.")

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left side: Video & 3D ---
        left_panel = QVBoxLayout()

        # Video display
        self.video_label = QLabel("No Video Loaded", self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setMinimumSize(480, 270)
        left_panel.addWidget(self.video_label, stretch=1)

        # 3D pose display
        self.openGLWidget = OpenGLWidget(self)
        self.openGLWidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.openGLWidget.setMinimumSize(480, 270)
        self.openGLWidget.statusUpdateRequest.connect(self.show_status_message) # Connect signal
        left_panel.addWidget(self.openGLWidget, stretch=1)

        # Frame navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("< Previous", self)
        self.prev_button.clicked.connect(self.prev_frame)
        nav_layout.addWidget(self.prev_button)

        self.frame_label = QLabel(f"Frame: {self.frame_index} / {self.total_frames -1}", self)
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(self.frame_label, stretch=1) # Allow label to expand

        self.next_button = QPushButton("Next >", self)
        self.next_button.clicked.connect(self.next_frame)
        nav_layout.addWidget(self.next_button)
        left_panel.addLayout(nav_layout)


        main_layout.addLayout(left_panel, stretch=2) # Give left panel more stretch


        # --- Right side: Controls & Labels ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # File Loading Section
        file_layout = QFormLayout()
        self.load_video_button = QPushButton("Load Video (.mp4)", self)
        self.load_video_button.clicked.connect(self.load_video)
        self.load_keypoints_button = QPushButton("Load Keypoints (.npy, .csv)", self)
        self.load_keypoints_button.clicked.connect(self.load_keypoints)
        self.video_path_label = QLabel("None", self)
        self.keypoints_path_label = QLabel("None", self)
        file_layout.addRow(self.load_video_button, self.video_path_label)
        file_layout.addRow(self.load_keypoints_button, self.keypoints_path_label)
        right_panel.addLayout(file_layout)

         # ID Section
        id_layout = QFormLayout()
        self.climber_id_input = QLineEdit("climber_001", self)
        self.route_id_input = QLineEdit("route_001", self)
        id_layout.addRow("Climber ID:", self.climber_id_input)
        id_layout.addRow("Route ID:", self.route_id_input)
        right_panel.addLayout(id_layout)

        # Label Control Section
        label_controls_layout = QHBoxLayout()
        self.load_labels_button = QPushButton("Load Label Names (.txt)", self)
        self.load_labels_button.clicked.connect(self.load_label_names_and_init_data) # Combined action
        # Add Label button might not be needed if loading from file is primary method?
        # self.add_label_button = QPushButton("Add New Label Type", self)
        # self.add_label_button.clicked.connect(self.add_new_label_type_manually) # Requires manual name input?
        self.save_button = QPushButton("Save Labels to CSV", self)
        self.save_button.clicked.connect(self.save_csv)
        label_controls_layout.addWidget(self.load_labels_button)
        # label_controls_layout.addWidget(self.add_label_button)
        label_controls_layout.addWidget(self.save_button)
        right_panel.addLayout(label_controls_layout)


        # Labels Area
        right_panel.addWidget(QLabel("--- Labels (Current Frame Values) ---"), alignment=Qt.AlignmentFlag.AlignCenter)
        self.labels_scroll_area = QScrollArea(self)
        self.labels_scroll_area.setWidgetResizable(True)
        self.labels_scroll_area.setStyleSheet("background-color: #ffffff;")
        self.labels_widget = QWidget()
        self.labels_layout = QVBoxLayout(self.labels_widget) # Main layout for all label rows
        self.labels_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.labels_scroll_area.setWidget(self.labels_widget)
        right_panel.addWidget(self.labels_scroll_area, stretch=1)

        main_layout.addLayout(right_panel, stretch=1)

        self.show()

    # --- UI Update ---

    def update_widget_states(self):
        """ Enable/disable widgets based on loaded data. """
        has_video = self.cap is not None and self.cap.isOpened()
        has_keypoints = self.keypoints is not None
        has_labels = bool(self.label_names) # Check if label names are loaded

        self.prev_button.setEnabled(has_keypoints or has_video)
        self.next_button.setEnabled(has_keypoints or has_video)
        self.openGLWidget.setEnabled(has_keypoints)
        self.save_button.setEnabled(has_keypoints and has_labels and self.csv_file is not None)

        # Enable label value inputs only if keypoints are loaded
        for elements in self.label_ui_elements.values():
            elements.value_widget.setEnabled(has_keypoints)
            # Maybe disable name/numeric checkbox after initial load?
            # elements.name_widget.setEnabled(False)
            # elements.numeric_checkbox.setEnabled(False)

        if not (has_keypoints or has_video):
             self.frame_index = 0
             self.frame_label.setText("Frame: N/A")
        else:
             self.update_frame_display() # Update label based on current index/total


    def show_status_message(self, message, timeout=3000):
        self.statusBar.showMessage(message, timeout)

    def display_error_message(self, title, message):
        QMessageBox.critical(self, title, message)
        self.show_status_message(f"Error: {message}", 5000)


    # --- File Loading & Data Initialization ---

    def load_video(self):
        # Uses same logic as ClassicInterface.load_video
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
                else:
                    self.total_frames = num_video_frames

                self.frame_index = 0
                self.update_widget_states()
                self.display_video_frame()
                self.show_status_message(f"Video loaded: {os.path.basename(video_file)} ({self.total_frames} frames)", 5000)

            except Exception as e:
                self.display_error_message("Video Load Error", f"Failed to load video: {e}")
                self.cap = None; self.video_path = None; self.video_path_label.setText("None")
                self.total_frames = self.keypoints.shape[0] if self.keypoints is not None else 0
                self.update_widget_states()

    def load_keypoints(self):
        # Uses similar logic to ClassicInterface.load_keypoints
        file_filter = "Keypoints Files (*.npy *.csv);;All Files (*)"
        keypoints_file, _ = QFileDialog.getOpenFileName(self, "Select Keypoints File", "", file_filter)
        if keypoints_file:
            keypoints_data = load_keypoint_data(keypoints_file) # Use updated util
            if keypoints_data is None:
                self.display_error_message("Keypoint Load Error", f"Failed to load keypoints from:\n{keypoints_file}")
                return

            if keypoints_data.ndim != 3 or keypoints_data.shape[2] != 3:
                 self.display_error_message("Keypoint Data Error", f"Invalid keypoints shape: {keypoints_data.shape}. Expected (frames, points, 3).")
                 return
            keypoints_data[:, :, 1] *= -1
            self.show_status_message("Note: Flipped Y-coordinate of keypoints for OpenGL display.",
                                     4000)  # Optional message

            self.keypoints = keypoints_data
            self.keypoints_path = keypoints_file
            self.keypoints_path_label.setText(os.path.basename(keypoints_file))
            num_keypoint_frames = self.keypoints.shape[0]

            if self.cap is not None and self.cap.isOpened():
                num_video_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if num_keypoint_frames != num_video_frames:
                     QMessageBox.warning(self, "Frame Count Mismatch", f"Keypoints: {num_keypoint_frames} frames, Video: {num_video_frames} frames. Using keypoint count.")
                self.total_frames = num_keypoint_frames
            else:
                self.total_frames = num_keypoint_frames

            self.frame_index = 0
            self.openGLWidget.set_data(self.keypoints, self.limbSeq)

            # Crucially, load/initialize data AFTER keypoints and labels are potentially loaded
            if self.label_names: # If labels already loaded, init data now
                self._load_or_initialize_label_data()
            else: # Otherwise, wait for Load Labels button
                 self.show_status_message("Keypoints loaded. Please load label names to proceed.", 5000)


            self.update_widget_states()
            self.display_video_frame() # Update video frame as well
            self.show_status_message(f"Keypoints loaded: {os.path.basename(keypoints_file)} ({self.total_frames} frames, {self.keypoints.shape[1]} points)", 5000)


    def load_label_names_and_init_data(self):
         """ Loads names from file, updates UI, then loads/initializes CSV data. """
         if self.keypoints is None:
              QMessageBox.warning(self, "Load Order", "Please load keypoints before loading label names.")
              return

         file_filter = "Text Files (*.txt);;All Files (*)"
         label_file, _ = QFileDialog.getOpenFileName(self, "Select Label Names File (comma-separated)", "", file_filter)
         if not label_file:
             return

         try:
             with open(label_file, 'r') as file:
                 content = file.read()
                 if content.startswith('\ufeff'): content = content[1:] # Handle BOM
                 loaded_names = [name.strip() for name in content.split(',') if name.strip()]

             if not loaded_names:
                  self.show_status_message("Warning: Label names file was empty.", 5000)
                  return

             # --- Update internal label structures ---
             # Clear existing UI and data (or merge? Clear is simpler)
             self._clear_labels()

             self.label_names = loaded_names
             for name in self.label_names:
                 self.label_is_numeric[name] = False # Default to non-numeric
                 self._add_label_ui(name)

             self.show_status_message(f"Loaded {len(self.label_names)} label names. Initializing data...", 3000)
             QApplication.processEvents() # Show message

             # --- Load or Initialize Data ---
             self._load_or_initialize_label_data()

             # Update UI based on loaded data for current frame
             self.update_label_value_inputs()
             self.update_widget_states()

         except FileNotFoundError:
             self.display_error_message("Load Error", f"Label names file not found:\n{label_file}")
         except Exception as e:
             self.display_error_message("Load Error", f"Failed to load or parse label names file:\n{e}")
             self._clear_labels() # Clear potentially inconsistent state
             self.update_widget_states()


    def _clear_labels(self):
         """ Clears label data structures and UI elements. """
         self.label_names = []
         self.label_is_numeric = {}
         self.label_values = {}
         self.csv_file = None
         self._has_unsaved_changes = False

         # Clear UI elements from layout
         for label_name in list(self.label_ui_elements.keys()): # Iterate over keys copy
            elements = self.label_ui_elements.pop(label_name)
            if elements.name_widget.parentWidget(): # Check if part of a layout/widget
                elements.name_widget.parentWidget().deleteLater() # Delete the container widget holding the form row

         # Clear the main labels layout just in case
         while self.labels_layout.count():
              child = self.labels_layout.takeAt(0)
              if child.widget():
                   child.widget().deleteLater()


    def _add_label_ui(self, label_name):
         """ Adds UI row for a given label name. """
         if label_name in self.label_ui_elements:
             print(f"Warning: UI for label '{label_name}' already exists.")
             return

         label_widget = QWidget(self) # Container for the form layout row
         label_form = QFormLayout(label_widget)
         label_form.setContentsMargins(0, 0, 0, 0)
         label_form.setSpacing(5)

         # Use QLabel for name (read-only after loading)
         label_name_display = QLabel(label_name)
         label_value_input = QLineEdit(self)
         label_value_input.setEnabled(False) # Disabled until keypoints loaded
         numeric_label_checkbox = QCheckBox("Numeric", self)

         label_form.addRow(label_name_display, label_value_input)
         label_form.addRow(numeric_label_checkbox) # Add checkbox below for better layout

         self.labels_layout.addWidget(label_widget)

         # Store UI elements
         ui_elements = LabelUIElements(label_name_display, label_value_input, numeric_label_checkbox)
         self.label_ui_elements[label_name] = ui_elements

         # Connect signals
         # Update numeric status in data
         numeric_label_checkbox.stateChanged.connect(
             lambda state, name=label_name: self.label_is_numeric.update({name: state == Qt.CheckState.Checked.value})
         )
         # Update label_values dict when input changes
         label_value_input.textChanged.connect(
             lambda text, name=label_name, input_widget=label_value_input, num_cb=numeric_label_checkbox: self._update_label_data_from_input(name, text, input_widget, num_cb)
         )

         # Set initial state based on data (if data loaded before UI)
         numeric_label_checkbox.setChecked(self.label_is_numeric.get(label_name, False))
         value = self.label_values.get(self.frame_index, {}).get(label_name, "")
         label_value_input.setText(str(value))


    def _load_or_initialize_label_data(self):
        """
        Loads label data from a CSV file if it exists, matching the loaded keypoints.
        Otherwise, initializes an empty data structure and creates a new CSV.
        This should be called AFTER keypoints and label names are loaded.
        """
        if self.keypoints is None or not self.label_names:
            self.show_status_message("Cannot initialize data: Load keypoints and label names first.", 5000)
            return

        # Determine expected CSV filename based on keypoints path
        if not self.keypoints_path:
             self.display_error_message("Data Error", "Cannot determine CSV path: Keypoints path is missing.")
             return
        base_name = os.path.splitext(self.keypoints_path)[0]
        self.csv_file = f"{base_name}_newlabels.csv" # Use distinct suffix

        self.label_values = {} # Reset internal data store
        self._has_unsaved_changes = False

        try:
            if os.path.exists(self.csv_file):
                self.show_status_message(f"Loading existing labels from {self.csv_file}...", 0)
                df = pd.read_csv(self.csv_file)

                # Validate essential columns exist
                required_cols = ["frame"] + self.label_names
                missing_cols = [col for col in required_cols if col not in df.columns]
                if "frame" not in df.columns:
                    raise ValueError("Existing CSV is missing the required 'frame' column.")

                if missing_cols:
                    self.show_status_message(f"Warning: CSV missing columns for labels: {missing_cols}. They will be initialized with defaults.", 5000)
                    # Add missing label columns to DataFrame with default values before processing
                    for col in missing_cols:
                        is_numeric = self.label_is_numeric.get(col, False)
                        df[col] = 0.0 if is_numeric else ""

                # Populate internal dictionary
                for _, row in df.iterrows():
                    frame = int(row["frame"])
                    if frame not in self.label_values:
                        self.label_values[frame] = {}
                    for label_name in self.label_names:
                        value = row.get(label_name) # Use .get for safety
                        # Convert type based on checkbox (which reflects self.label_is_numeric)
                        is_numeric = self.label_is_numeric.get(label_name, False)
                        if pd.isna(value): # Handle NaN/None from CSV
                            value = 0.0 if is_numeric else ""
                        else:
                            if is_numeric:
                                try: value = float(value)
                                except (ValueError, TypeError): value = 0.0 # Default on conversion error
                            else:
                                value = str(value) # Ensure string for non-numeric
                        self.label_values[frame][label_name] = value

                self.show_status_message(f"Successfully loaded labels from {self.csv_file}", 5000)

            else:
                self.show_status_message(f"No existing label file found. Creating new file: {self.csv_file}", 3000)
                # Initialize empty internal dictionary with defaults for all frames
                num_frames = self.keypoints.shape[0]
                for frame_idx in range(num_frames):
                    self.label_values[frame_idx] = {}
                    for label_name in self.label_names:
                        is_numeric = self.label_is_numeric.get(label_name, False)
                        self.label_values[frame_idx][label_name] = 0.0 if is_numeric else ""
                # Optionally, save this initial empty/default CSV immediately?
                # self.save_csv(show_dialog=False) # Save without asking for path again
                # Or just let the first explicit save create it. Let's do the latter.
                self.show_status_message(f"Initialized empty labels for {num_frames} frames. Save to create CSV.", 5000)

        except FileNotFoundError:
             # This case should technically be handled by os.path.exists, but include for safety
             self.display_error_message("Load Error", f"Could not find CSV file: {self.csv_file}")
             self.csv_file = None # Reset path if load failed
        except (ValueError, KeyError, pd.errors.ParserError, Exception) as e:
             self.display_error_message("Load Error", f"Failed to load or parse existing CSV ({self.csv_file}):\n{e}\n\nRe-initializing with empty labels.")
             # Reset state to empty on error
             self.label_values = {}
             self.csv_file = f"{base_name}_newlabels.csv" # Keep intended path, but data is lost
             num_frames = self.keypoints.shape[0]
             for frame_idx in range(num_frames):
                  self.label_values[frame_idx] = {name: (0.0 if self.label_is_numeric.get(name, False) else "") for name in self.label_names}

        # Final step: Update UI to reflect data for current frame
        self.update_label_value_inputs()
        self.update_widget_states() # Re-enable save button if successful

    # --- Frame Navigation & Display ---

    def prev_frame(self):
        if self.total_frames > 0 and self.frame_index > 0:
            self.frame_index -= 1
            self.update_frame_display()

    def next_frame(self):
        if self.total_frames > 0 and self.frame_index < self.total_frames - 1:
            self.frame_index += 1
            self.update_frame_display()

    def update_frame_display(self):
        """ Updates frame label, video, 3D pose, and label input values. """
        if not (self.keypoints is not None or self.cap is not None):
            self.frame_label.setText("Frame: N/A")
            return

        self.frame_label.setText(f"Frame: {self.frame_index} / {self.total_frames - 1 if self.total_frames > 0 else 0}")

        self.display_video_frame()
        if self.keypoints is not None:
             self.openGLWidget.set_frame_index(self.frame_index)
        else:
             self.openGLWidget.update()

        # Update label input fields to show values for the new frame
        self.update_label_value_inputs()


    def display_video_frame(self):
        # Uses same logic as ClassicInterface.display_video_frame
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            ret, frame = self.cap.read()
            if ret:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    self.video_label.setPixmap(scaled_pixmap)
                except Exception as e:
                    print(f"Error converting/displaying frame {self.frame_index}: {e}")
                    self.video_label.setText("Error Displaying Frame")
            else:
                self.video_label.setText(f"End of Video or Read Error (Frame {self.frame_index})")
        else:
             self.video_label.setText("No Video Loaded")

    # --- Label Value Handling ---

    def update_label_value_inputs(self):
        """ Sets the text in the QLineEdit widgets based on self.label_values for the current frame. """
        current_frame_values = self.label_values.get(self.frame_index, {})
        for label_name, elements in self.label_ui_elements.items():
            value = current_frame_values.get(label_name, "") # Default to empty string if missing
            # Block signals temporarily to prevent update loop when setting text
            elements.value_widget.blockSignals(True)
            elements.value_widget.setText(str(value))
            elements.value_widget.blockSignals(False)
            # Also update checkbox state, though it shouldn't change per frame
            elements.numeric_checkbox.blockSignals(True)
            elements.numeric_checkbox.setChecked(self.label_is_numeric.get(label_name, False))
            elements.numeric_checkbox.blockSignals(False)


    def _update_label_data_from_input(self, label_name, text, input_widget, numeric_checkbox):
        """ Updates self.label_values when a QLineEdit's text changes. """
        if self.frame_index not in self.label_values:
            self.label_values[self.frame_index] = {}

        is_numeric = numeric_checkbox.isChecked()
        current_value = None
        parse_error = False

        if is_numeric:
            try:
                current_value = float(text.strip()) # Store numeric as float
            except ValueError:
                # Handle invalid numeric input - visual feedback?
                input_widget.setStyleSheet("QLineEdit { background-color: #ffdddd; }") # Indicate error
                self.show_status_message(f"Warning: Invalid numeric input '{text}' for {label_name}.", 3000)
                parse_error = True
                current_value = 0.0 # Or keep previous valid value? Using 0.0 for now.
            else:
                # Valid input, reset style
                input_widget.setStyleSheet("") # Reset to default
        else:
            current_value = text # Store as string
            input_widget.setStyleSheet("") # Reset style if previously error

        # Only update data if parsing was successful or not numeric
        if not parse_error:
             if self.label_values[self.frame_index].get(label_name) != current_value:
                 self.label_values[self.frame_index][label_name] = current_value
                 self._has_unsaved_changes = True # Mark changes as unsaved


    # --- Saving ---

    def save_csv(self, show_dialog=True):
        """ Saves the current label data (all frames) to the associated CSV file. """
        if self.keypoints is None:
            self.display_error_message("Save Error", "No keypoints loaded.")
            return
        if not self.label_names:
            self.display_error_message("Save Error", "No labels defined.")
            return
        if not self.csv_file:
            # This shouldn't happen if _load_or_initialize ran correctly
             self.display_error_message("Save Error", "CSV file path not set. Try loading labels again.")
             # Optionally, prompt user to choose a save path now?
             show_dialog = True # Force dialog if path is missing

        save_path = self.csv_file
        if show_dialog:
             # Allow user to confirm or change save location
            suggested_path = self.csv_file if self.csv_file else "keypoints_newlabels.csv"
            save_path_selected, _ = QFileDialog.getSaveFileName(self, "Save Labels CSV", suggested_path, "CSV Files (*.csv)")
            if not save_path_selected:
                 self.show_status_message("Save cancelled.", 2000)
                 return
            save_path = save_path_selected
            self.csv_file = save_path # Update the associated path

        climber_id = self.climber_id_input.text().strip()
        route_id = self.route_id_input.text().strip()

        self.show_status_message(f"Saving data to {save_path}...")
        QApplication.processEvents() # Allow UI update

        data_to_save = []
        num_frames = self.keypoints.shape[0]
        processed_count = 0

        try:
            # Prepare headers
            headers = ["climber_id", "route_id", "frame"]
            num_kps = self.keypoints.shape[1]
            for i in range(num_kps):
                 headers.extend([f"kp{i}_x", f"kp{i}_y", f"kp{i}_z"])
            headers.extend(self.label_names) # Add loaded label names

            for frame_idx in range(num_frames):
                frame_data = {}
                frame_data["climber_id"] = climber_id
                frame_data["route_id"] = route_id
                frame_data["frame"] = frame_idx

                # Add keypoints
                for kp_idx, point in enumerate(self.keypoints[frame_idx]):
                    frame_data[f"kp{kp_idx}_x"] = point[0]
                    frame_data[f"kp{kp_idx}_y"] = point[1]
                    frame_data[f"kp{kp_idx}_z"] = point[2]

                # Add labels from internal dictionary
                frame_labels = self.label_values.get(frame_idx, {}) # Get labels for this frame
                for label_name in self.label_names:
                    is_numeric = self.label_is_numeric.get(label_name, False)
                    default_value = 0.0 if is_numeric else ""
                    # Get value, provide default if label/frame missing
                    value = frame_labels.get(label_name, default_value)
                    # Ensure type consistency before saving (redundant if _update handled it, but safe)
                    if is_numeric:
                         try: value = float(value)
                         except (ValueError, TypeError): value = 0.0
                    else:
                         value = str(value)
                    frame_data[label_name] = value


                data_to_save.append(frame_data)
                processed_count += 1
                if processed_count % 100 == 0: # Update status periodically
                     self.show_status_message(f"Saving... Processed {processed_count}/{num_frames} frames.", 0)
                     QApplication.processEvents()

            # Create DataFrame and save
            df = pd.DataFrame(data_to_save, columns=headers) # Ensure column order
            df.to_csv(save_path, index=False)
            self._has_unsaved_changes = False # Mark as saved
            self.show_status_message(f"Successfully saved labels to {save_path}", 5000)

        except Exception as e:
            self.display_error_message("Save Error", f"An unexpected error occurred during saving:\n{e}")
            self.show_status_message("Save failed.", 5000)


    # --- Window Closing ---
    def closeEvent(self, event: QCloseEvent):
        # Optional: Check for unsaved changes
        if self._has_unsaved_changes:
             reply = QMessageBox.question(self, 'Unsaved Changes',
                                           "You have unsaved changes. Save before closing?",
                                           QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                                           QMessageBox.StandardButton.Cancel) # Default to Cancel

             if reply == QMessageBox.StandardButton.Save:
                 self.save_csv(show_dialog=False) # Save to current path without asking again
                 if self._has_unsaved_changes: # If save failed somehow, don't close
                      event.ignore()
                      return
             elif reply == QMessageBox.StandardButton.Cancel:
                 event.ignore() # Don't close
                 return
             # If Discard, proceed to close

        # Release video capture
        if self.cap:
            self.cap.release()
        event.accept() # Proceed with closing