import sys
import numpy as np
import cv2
import pandas as pd
from PyQt6.QtWidgets import (QMainWindow, QHBoxLayout, QVBoxLayout, QLabel,
                             QSlider, QWidget, QPushButton, QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap  # Ensure QImage is imported
from open_gl_widget import OpenGLWidget
from label_dialog import LabelDialog
from utils import load_keypoint_data

class PoseLabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.keypoints = None
        self.frame_index = 0
        self.limbSeq = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
        self.cap = None
        self.labels = []

        self.initUI()
        self.open_files()

    def initUI(self):
        self.setWindowTitle('3D Pose Labeling Tool')
        self.setGeometry(100, 100, 1200, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left side: Video display
        video_layout = QVBoxLayout()
        self.video_label = QLabel(self)
        video_layout.addWidget(self.video_label)
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
        self.openGLWidget.limbSeq = self.limbSeq
        pose_layout.addWidget(self.openGLWidget)

        self.save_button = QPushButton("Save CSV", self)
        self.save_button.clicked.connect(self.save_csv)
        pose_layout.addWidget(self.save_button)

        self.add_label_button = QPushButton("Add Label", self)
        self.add_label_button.clicked.connect(self.add_label)
        pose_layout.addWidget(self.add_label_button)

        main_layout.addLayout(pose_layout)

        self.show()

    def open_files(self):
        npy_file, _ = QFileDialog.getOpenFileName(self, "Select Keypoints File", "", "NPY Files (*.npy);;All Files (*)")
        if npy_file:
            self.keypoints = load_keypoint_data(npy_file)
            self.slider.setMaximum(self.keypoints.shape[0] - 1)
            self.openGLWidget.keypoints = self.keypoints

        video_file, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "MP4 Files (*.mp4);;All Files (*)")
        if video_file:
            self.cap = cv2.VideoCapture(video_file)
            if not self.cap.isOpened():
                raise ValueError("Error: Could not open video.")

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
                self.video_label.setPixmap(QPixmap.fromImage(image))

    def add_label(self):
        dialog = LabelDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            label_name, label_value, start_frame, end_frame = dialog.get_values()
            self.labels.append((label_name, label_value, start_frame, end_frame))

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
            for label in self.labels:
                label_name, label_value, start_frame, end_frame = label
                if start_frame <= frame <= end_frame:
                    frame_data[label_name] = label_value
            data.append(frame_data)

        df = pd.DataFrame(data)
        df.to_csv("keypoints_labels.csv", index=False)
        print("CSV saved successfully.")
