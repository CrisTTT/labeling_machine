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

class ClassicInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.keypoints = None
        self.frame_index = 0
        self.limbSeq = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
        self.cap = None
        self.labels = []

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Classic 3D Pose Labeling Tool')
        self.setGeometry(100, 100, 1200, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: Video display
        video_layout = QVBoxLayout()
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(600, 400)  # Fixed size for video display
        video_layout.addWidget(self.video_label)
        self.load_video_button = QPushButton("Load Video", self)
        self.load_video_button.clicked.connect(self.load_video)
        video_layout.addWidget(self.load_video_button)
        main_layout.addLayout(video_layout)

        # Right side: 3D pose display and controls
        pose_layout = QVBoxLayout()
        self.frame_label = QLabel(f"Frame: {self.frame_index}", self)
        pose_layout.addWidget(self.frame_label)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.update_frame)
        pose_layout.addWidget(self.slider)

        self.openGLWidget = OpenGLWidget(self)
        self.openGLWidget.setFixedSize(600, 400)  # Fixed size for 3D pose display
        self.openGLWidget.limbSeq = self.limbSeq
        pose_layout.addWidget(self.openGLWidget)

        self.save_button = QPushButton("Save CSV", self)
        self.save_button.clicked.connect(self.save_csv)
        pose_layout.addWidget(self.save_button)

        self.add_label_button = QPushButton("Add Label", self)
        self.add_label_button.clicked.connect(self.add_label)
        pose_layout.addWidget(self.add_label_button)

        self.load_labels_button = QPushButton("Load Label Names", self)
        self.load_labels_button.clicked.connect(self.load_label_names)
        pose_layout.addWidget(self.load_labels_button)

        self.load_keypoints_button = QPushButton("Load Keypoints", self)
        self.load_keypoints_button.clicked.connect(self.load_keypoints)
        pose_layout.addWidget(self.load_keypoints_button)

        # Labels layout with scroll area
        self.labels_scroll_area = QScrollArea(self)
        self.labels_scroll_area.setWidgetResizable(True)
        self.labels_widget = QWidget()
        self.labels_layout = QVBoxLayout(self.labels_widget)
        self.labels_scroll_area.setWidget(self.labels_widget)
        pose_layout.addWidget(self.labels_scroll_area)

        main_layout.addLayout(pose_layout)

        self.show()

    def load_video(self):
        video_file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "MP4 Files (*.mp4);;All Files (*)")
        if video_file:
            self.cap = cv2.VideoCapture(video_file)
            if not self.cap.isOpened():
                raise ValueError("Error: Could not open video.")
            self.slider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            self.display_video_frame()

    def load_keypoints(self):
        file_filter = "Keypoints Files (*.npy *.csv);;All Files (*)"
        keypoints_file, _ = QFileDialog.getOpenFileName(self, "Select Keypoints File", "", file_filter)
        if keypoints_file:
            self.keypoints = load_keypoint_data(keypoints_file)
            self.slider.setMaximum(self.keypoints.shape[0] - 1)
            self.openGLWidget.keypoints = self.keypoints

    def update_frame(self, value):
        try:
            self.frame_index = value
            self.frame_label.setText(f"Frame: {self.frame_index}")
            self.openGLWidget.frame_index = self.frame_index
            self.openGLWidget.update()
            self.display_video_frame()
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

    def add_label(self):
        try:
            # Create a new label entry
            label_name = QLineEdit(self)
            numeric_label = QCheckBox("Numeric Label", self)

            label_collapse_button = QToolButton(self)
            label_collapse_button.setArrowType(Qt.ArrowType.RightArrow)
            label_collapse_button.setCheckable(True)
            label_collapse_button.setChecked(False)
            label_collapse_button.setIconSize(QSize(12, 12))

            label_widget = QWidget(self)
            label_widget_layout = QVBoxLayout(label_widget)
            label_form = QFormLayout()
            label_form.addRow("Label Name:", label_name)
            label_form.addRow("Numeric Label:", numeric_label)
            label_widget_layout.addLayout(label_form)
            label_widget_layout.addWidget(label_collapse_button)

            values_widget = QWidget(self)
            values_layout = QVBoxLayout(values_widget)
            add_value_button = QPushButton("Add Value", self)
            values_layout.addWidget(add_value_button)

            label_widget_layout.addWidget(values_widget)
            self.labels_layout.addWidget(label_widget)

            # Store the label inputs and values
            values = []
            self.labels.append((label_name, numeric_label, values, values_widget, label_collapse_button))

            def add_value():
                value_widget = QWidget(self)
                value_layout = QVBoxLayout(value_widget)
                value_collapse_button = QToolButton(self)
                value_collapse_button.setArrowType(Qt.ArrowType.RightArrow)
                value_collapse_button.setCheckable(True)
                value_collapse_button.setChecked(False)
                value_collapse_button.setIconSize(QSize(12, 12))
                value_layout.addWidget(value_collapse_button)

                value_input = QLineEdit(self)
                value_layout.addWidget(value_input)

                intervals_widget = QWidget(self)
                intervals_layout = QVBoxLayout(intervals_widget)
                add_interval_button = QPushButton("Add Interval", self)
                intervals_layout.addWidget(add_interval_button)

                value_layout.addWidget(intervals_widget)
                values_layout.addWidget(value_widget)

                intervals = []
                values.append((value_input, intervals, value_collapse_button))

                def add_interval():
                    start_frame = QLineEdit(self)
                    end_frame = QLineEdit(self)
                    interval_layout = QFormLayout()
                    interval_layout.addRow("Start Frame:", start_frame)
                    interval_layout.addRow("End Frame:", end_frame)
                    intervals_layout.addLayout(interval_layout)
                    intervals.append((start_frame, end_frame))

                add_interval_button.clicked.connect(add_interval)

                def toggle_intervals():
                    try:
                        if value_collapse_button.isChecked():
                            value_collapse_button.setArrowType(Qt.ArrowType.DownArrow)
                            intervals_widget.setVisible(True)
                        else:
                            value_collapse_button.setArrowType(Qt.ArrowType.RightArrow)
                            intervals_widget.setVisible(False)
                    except Exception as e:
                        print(f"Error toggling intervals: {e}")

                value_collapse_button.clicked.connect(toggle_intervals)
                intervals_widget.setVisible(False)

            add_value_button.clicked.connect(add_value)

            def toggle_values():
                try:
                    if label_collapse_button.isChecked():
                        label_collapse_button.setArrowType(Qt.ArrowType.DownArrow)
                        values_widget.setVisible(True)
                    else:
                        label_collapse_button.setArrowType(Qt.ArrowType.RightArrow)
                        values_widget.setVisible(False)
                except Exception as e:
                    print(f"Error toggling values: {e}")

            label_collapse_button.clicked.connect(toggle_values)
            values_widget.setVisible(False)

        except Exception as e:
            print(f"Error adding label: {e}")

    def load_label_names(self):
        file_filter = "Text Files (*.txt);;All Files (*)"
        label_file, _ = QFileDialog.getOpenFileName(self, "Select Label Names File", "", file_filter)
        if label_file:
            with open(label_file, 'r') as file:
                label_names = file.read().split(',')
                for label_name in label_names:
                    self.add_label_from_name(label_name.strip())

    def add_label_from_name(self, label_name):
        numeric_label = QCheckBox("Numeric Label", self)

        label_collapse_button = QToolButton(self)
        label_collapse_button.setArrowType(Qt.ArrowType.RightArrow)
        label_collapse_button.setCheckable(True)
        label_collapse_button.setChecked(False)
        label_collapse_button.setIconSize(QSize(12, 12))

        label_widget = QWidget(self)
        label_widget_layout = QVBoxLayout(label_widget)
        label_form = QFormLayout()
        label_form.addRow("Label Name:", QLineEdit(label_name, self))
        label_form.addRow("Numeric Label:", numeric_label)
        label_widget_layout.addLayout(label_form)
        label_widget_layout.addWidget(label_collapse_button)

        values_widget = QWidget(self)
        values_layout = QVBoxLayout(values_widget)
        add_value_button = QPushButton("Add Value", self)
        values_layout.addWidget(add_value_button)

        label_widget_layout.addWidget(values_widget)
        self.labels_layout.addWidget(label_widget)

        values = []
        self.labels.append((QLineEdit(label_name, self), numeric_label, values, values_widget, label_collapse_button))

        def add_value():
            value_widget = QWidget(self)
            value_layout = QVBoxLayout(value_widget)
            value_collapse_button = QToolButton(self)
            value_collapse_button.setArrowType(Qt.ArrowType.RightArrow)
            value_collapse_button.setCheckable(True)
            value_collapse_button.setChecked(False)
            value_collapse_button.setIconSize(QSize(12, 12))
            value_layout.addWidget(value_collapse_button)

            value_input = QLineEdit(self)
            value_layout.addWidget(value_input)

            intervals_widget = QWidget(self)
            intervals_layout = QVBoxLayout(intervals_widget)
            add_interval_button = QPushButton("Add Interval", self)
            intervals_layout.addWidget(add_interval_button)

            value_layout.addWidget(intervals_widget)
            values_layout.addWidget(value_widget)

            intervals = []
            values.append((value_input, intervals, value_collapse_button))

            def add_interval():
                start_frame = QLineEdit(self)
                end_frame = QLineEdit(self)
                interval_layout = QFormLayout()
                interval_layout.addRow("Start Frame:", start_frame)
                interval_layout.addRow("End Frame:", end_frame)
                intervals_layout.addLayout(interval_layout)
                intervals.append((start_frame, end_frame))

            add_interval_button.clicked.connect(add_interval)

            def toggle_intervals():
                try:
                    if value_collapse_button.isChecked():
                        value_collapse_button.setArrowType(Qt.ArrowType.DownArrow)
                        intervals_widget.setVisible(True)
                    else:
                        value_collapse_button.setArrowType(Qt.ArrowType.RightArrow)
                        intervals_widget.setVisible(False)
                except Exception as e:
                    print(f"Error toggling intervals: {e}")

            value_collapse_button.clicked.connect(toggle_intervals)
            intervals_widget.setVisible(False)

        add_value_button.clicked.connect(add_value)

        def toggle_values():
            try:
                if label_collapse_button.isChecked():
                    label_collapse_button.setArrowType(Qt.ArrowType.DownArrow)
                    values_widget.setVisible(True)
                else:
                    label_collapse_button.setArrowType(Qt.ArrowType.RightArrow)
                    values_widget.setVisible(False)
            except Exception as e:
                print(f"Error toggling values: {e}")

        label_collapse_button.clicked.connect(toggle_values)
        values_widget.setVisible(False)

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
            for label_name, numeric_label, values, _, _ in self.labels:
                label_name_text = label_name.text()
                is_numeric = numeric_label.isChecked()
                for value_input, intervals, _ in values:
                    for start_frame, end_frame in intervals:
                        if int(start_frame.text()) <= frame <= int(end_frame.text()):
                            frame_data[label_name_text] = value_input.text()
                            break
                    else:
                        if is_numeric:
                            frame_data[label_name_text] = 0
            data.append(frame_data)

        df = pd.DataFrame(data)
        df.to_csv("keypoints_labels.csv", index=False)
        print("CSV saved successfully.")
