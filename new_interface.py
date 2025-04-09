import sys
import numpy as np
import cv2
import pandas as pd
from PyQt6.QtWidgets import (QMainWindow, QHBoxLayout, QVBoxLayout, QLabel,
                             QSlider, QWidget, QPushButton, QFileDialog, QLineEdit, QFormLayout, QCheckBox, QToolButton, QScrollArea)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QIcon
from open_gl_widget import OpenGLWidget
from utils import load_keypoint_data
import os
from datetime import datetime

class NewInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.keypoints = None
        self.frame_index = 0
        self.limbSeq = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
        self.cap = None
        self.labels = []
        self.label_values = {}  # Dictionary to store label values for each frame
        self.csv_file = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('New 3D Pose Labeling Tool')
        self.setGeometry(100, 100, 1200, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: Video and 3D pose display
        left_layout = QVBoxLayout()

        # Video display
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(600, 300)  # Fixed size for video display
        left_layout.addWidget(self.video_label)

        # 3D pose display
        self.openGLWidget = OpenGLWidget(self)
        self.openGLWidget.setFixedSize(600, 300)  # Fixed size for 3D pose display
        self.openGLWidget.limbSeq = self.limbSeq
        left_layout.addWidget(self.openGLWidget)

        # Frame navigation buttons
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous", self)
        self.prev_button.clicked.connect(self.prev_frame)
        nav_layout.addWidget(self.prev_button)

        self.frame_label = QLabel(f"Frame: {self.frame_index}", self)
        nav_layout.addWidget(self.frame_label)

        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.next_frame)
        nav_layout.addWidget(self.next_button)

        left_layout.addLayout(nav_layout)

        main_layout.addLayout(left_layout)

        # Right side: Labels display
        right_layout = QVBoxLayout()

        self.load_video_button = QPushButton("Load Video", self)
        self.load_video_button.clicked.connect(self.load_video)
        right_layout.addWidget(self.load_video_button)

        self.load_keypoints_button = QPushButton("Load Keypoints", self)
        self.load_keypoints_button.clicked.connect(self.load_keypoints)
        right_layout.addWidget(self.load_keypoints_button)

        self.load_labels_button = QPushButton("Load Label Names", self)
        self.load_labels_button.clicked.connect(self.load_label_names)
        right_layout.addWidget(self.load_labels_button)

        self.save_button = QPushButton("Save CSV", self)
        self.save_button.clicked.connect(self.save_csv)
        right_layout.addWidget(self.save_button)

        # Labels layout with scroll area
        self.labels_scroll_area = QScrollArea(self)
        self.labels_scroll_area.setWidgetResizable(True)
        self.labels_widget = QWidget()
        self.labels_layout = QVBoxLayout(self.labels_widget)
        self.labels_scroll_area.setWidget(self.labels_widget)
        right_layout.addWidget(self.labels_scroll_area)

        main_layout.addLayout(right_layout)

        self.show()

    def load_video(self):
        video_file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "MP4 Files (*.mp4);;All Files (*)")
        if video_file:
            self.cap = cv2.VideoCapture(video_file)
            if not self.cap.isOpened():
                raise ValueError("Error: Could not open video.")
            self.frame_index = 0
            self.display_video_frame()

    def load_keypoints(self):
        file_filter = "Keypoints Files (*.npy *.csv);;All Files (*)"
        keypoints_file, _ = QFileDialog.getOpenFileName(self, "Select Keypoints File", "", file_filter)
        if keypoints_file:
            self.keypoints = load_keypoint_data(keypoints_file)
            self.frame_index = 0
            self.openGLWidget.keypoints = self.keypoints
            self.display_video_frame()
            self.initialize_csv()

    def load_label_names(self):
        file_filter = "Text Files (*.txt);;All Files (*)"
        label_file, _ = QFileDialog.getOpenFileName(self, "Select Label Names File", "", file_filter)
        if label_file:
            try:
                with open(label_file, 'r') as file:
                    label_names = file.read().split(',')
                    for label_name in label_names:
                        self.add_label_from_name(label_name.strip())
            except Exception as e:
                print(f"Error loading label names: {e}")

    def prev_frame(self):
        if self.frame_index > 0:
            self.frame_index -= 1
            self.update_frame()

    def next_frame(self):
        if self.cap and self.frame_index < int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1:
            self.frame_index += 1
            self.update_frame()

    def update_frame(self):
        try:
            self.frame_label.setText(f"Frame: {self.frame_index}")
            self.openGLWidget.frame_index = self.frame_index
            self.openGLWidget.update()
            self.display_video_frame()
            self.update_label_values()
        except Exception as e:
            print(f"Error updating frame: {e}")

    def display_video_frame(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(image).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def add_label_from_name(self, label_name):
        label_name_input = QLineEdit(label_name, self)
        label_value_input = QLineEdit(self)
        numeric_label = QCheckBox("Numeric Label", self)

        label_widget = QWidget(self)
        label_widget_layout = QVBoxLayout(label_widget)
        label_form = QFormLayout()
        label_form.addRow("Label Name:", label_name_input)
        label_form.addRow("Label Value:", label_value_input)
        label_form.addRow("Numeric Label:", numeric_label)
        label_widget_layout.addLayout(label_form)

        self.labels_layout.addWidget(label_widget)

        # Store the label inputs and values
        self.labels.append((label_name_input, label_value_input, numeric_label))

        # Initialize label values for the current frame
        if self.frame_index not in self.label_values:
            self.label_values[self.frame_index] = {}
        self.label_values[self.frame_index][label_name] = 0 if numeric_label.isChecked() else ""

        # Connect the value input to update the label values
        label_value_input.textChanged.connect(self.update_label_value)

    def update_label_value(self):
        for label_name_input, label_value_input, numeric_label in self.labels:
            label_name = label_name_input.text()
            if self.frame_index not in self.label_values:
                self.label_values[self.frame_index] = {}
            self.label_values[self.frame_index][label_name] = int(label_value_input.text()) if numeric_label.isChecked() else label_value_input.text()

    def update_label_values(self):
        for label_name_input, label_value_input, numeric_label in self.labels:
            label_name = label_name_input.text()
            if self.frame_index in self.label_values and label_name in self.label_values[self.frame_index]:
                value = self.label_values[self.frame_index][label_name]
                label_value_input.setText(str(value))

    def save_csv(self):
        climber_id = "climber_001"  # Example climber ID
        route_id = "route_001"  # Example route ID

        data = []
        for frame in range(self.keypoints.shape[0]):
            frame_data = {"climber_id": climber_id, "route_id": route_id, "frame": frame}
            for i, point in enumerate(self.keypoints[frame]):
                frame_data[f"kp{i}_x"] = point[0]
                frame_data[f"kp{i}_y"] = point[1]
                frame_data[f"kp{i}_z"] = point[2]
            for label_name_input, label_value_input, numeric_label in self.labels:
                label_name_text = label_name_input.text()
                is_numeric = numeric_label.isChecked()
                frame_data[label_name_text] = self.label_values.get(frame, {}).get(label_name_text, 0 if is_numeric else "")
            data.append(frame_data)

        df = pd.DataFrame(data)
        df.to_csv(self.csv_file, index=False)
        print("CSV saved successfully.")

    def initialize_csv(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = f"keypoints_labels_{timestamp}.csv"
        if not os.path.exists(self.csv_file):
            # Create an empty DataFrame with the necessary columns
            columns = ["climber_id", "route_id", "frame"] + [f"kp{i}_{axis}" for i in range(self.keypoints.shape[1]) for axis in "xyz"]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.csv_file, index=False)

        # Load the existing CSV to initialize label values
        if os.path.exists(self.csv_file):
            df = pd.read_csv(self.csv_file)
            for _, row in df.iterrows():
                frame = row["frame"]
                self.label_values[frame] = {label_name_input.text(): row[label_name_input.text()] for label_name_input, _, _ in self.labels}
