from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import warnings

class OpenGLWidget(QOpenGLWidget):
    """
    Widget for rendering 3D pose using OpenGL.
    Handles mouse rotation and displays keypoints and limbs.
    """
    # Signal to request status bar updates from the main window
    statusUpdateRequest = pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.zoom_factor = 5.0 # Initial distance/zoom
        self.last_pos = QPoint()

        # Data related attributes
        self.frame_index = 0
        self.keypoints = None # Expected shape: (n_frames, n_points, 3)
        self.limbSeq = None   # Expected format: list of [start_idx, end_idx]

        # Set focus policy to accept keyboard events if needed later
        # self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_data(self, keypoints, limbSeq):
        """ Safely sets keypoint and limb sequence data. """
        if keypoints is not None:
             # Basic validation of keypoints structure
            if not isinstance(keypoints, np.ndarray) or keypoints.ndim != 3 or keypoints.shape[2] != 3:
                self.statusUpdateRequest.emit(f"Error: Invalid keypoints data shape. Expected (frames, points, 3), got {keypoints.shape if isinstance(keypoints, np.ndarray) else type(keypoints)}.", 5000)
                self.keypoints = None
            else:
                self.keypoints = keypoints
                self.statusUpdateRequest.emit(f"Keypoints loaded. Shape: {self.keypoints.shape}", 3000)
        else:
             self.keypoints = None

        if limbSeq is not None:
            # Basic validation of limb sequence
             if isinstance(limbSeq, list) and all(isinstance(l, list) and len(l) == 2 for l in limbSeq):
                 self.limbSeq = limbSeq
             else:
                 self.statusUpdateRequest.emit("Error: Invalid limb sequence format. Expected list of [start, end] index pairs.", 5000)
                 self.limbSeq = None
        else:
            self.limbSeq = None

        self.update() # Trigger repaint

    def set_frame_index(self, index):
        """ Safely sets the current frame index. """
        if self.keypoints is not None:
            if 0 <= index < self.keypoints.shape[0]:
                self.frame_index = index
            else:
                # Index out of bounds, clamp or warn
                self.frame_index = max(0, min(index, self.keypoints.shape[0] - 1))
                self.statusUpdateRequest.emit(f"Warning: Frame index {index} out of bounds ({self.keypoints.shape[0]} frames). Clamped to {self.frame_index}.", 3000)
        else:
            self.frame_index = index # Allow setting even if no keypoints, but drawing will fail safely
        self.update() # Trigger repaint

    def initializeGL(self):
        """ Called once when the widget is initialized. """
        glClearColor(1.0, 1.0, 1.0, 1.0) # White background
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH) # Make points circular
        glEnable(GL_LINE_SMOOTH) # Antialias lines
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_BLEND) # Enable blending for antialiasing
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Initial camera position - move it back along Z-axis
        # This value might need adjustment based on keypoint data scale
        glTranslatef(0.0, 0.0, -self.zoom_factor)

    def resizeGL(self, w, h):
        """ Called when the widget is resized. """
        if h == 0: h = 1 # Prevent division by zero
        glViewport(0, 0, w, h)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        # Field of view, aspect ratio, near clip, far clip
        # Adjust near/far clips based on data scale if needed
        aspect_ratio = w / h
        gluPerspective(45.0, aspect_ratio, 0.1, 100.0)

        # Switch back to ModelView matrix for transformations
        glMatrixMode(GL_MODELVIEW)
        # No need to load identity here, paintGL will handle it

    def paintGL(self):
        """ Called whenever the widget needs to be painted. """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Reset ModelView matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # 1. Apply camera transform (move back)
        glTranslatef(0.0, 0.0, -self.zoom_factor)

        # 2. Apply mouse rotations
        glRotatef(self.rotation_x, 1.0, 0.0, 0.0) # Rotate around X-axis
        glRotatef(self.rotation_y, 0.0, 1.0, 0.0) # Rotate around Y-axis

        # 3. Draw scene elements
        self.draw_axes()
        self.draw_pose()

    def draw_axes(self):
        """ Draws X, Y, Z axes. """
        glLineWidth(1.5)
        glBegin(GL_LINES)
        # X axis (Red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        # Y axis (Green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        # Z axis (Blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.0)
        glEnd()

    def draw_pose(self):
        """ Draws the pose (keypoints and limbs) for the current frame. """
        if self.keypoints is None:
            # self.statusUpdateRequest.emit("Cannot draw pose: Keypoints not loaded.", 3000)
            return # Nothing to draw

        if not (0 <= self.frame_index < self.keypoints.shape[0]):
            self.statusUpdateRequest.emit(f"Error drawing pose: Frame index {self.frame_index} out of bounds.", 5000)
            return

        current_frame_keypoints = self.keypoints[self.frame_index]
        num_points = current_frame_keypoints.shape[0]

        # --- Draw Limbs ---
        if self.limbSeq is not None:
            glLineWidth(2.5)
            glColor3f(0.0, 0.0, 0.0) # Black lines
            glBegin(GL_LINES)
            for limb in self.limbSeq:
                try:
                    start_idx, end_idx = limb
                    if 0 <= start_idx < num_points and 0 <= end_idx < num_points:
                        start_point = current_frame_keypoints[start_idx]
                        end_point = current_frame_keypoints[end_idx]
                        # Ensure points are 3D
                        if len(start_point) == 3 and len(end_point) == 3:
                             glVertex3fv(start_point)
                             glVertex3fv(end_point)
                        else:
                             warnings.warn(f"Limb {limb}: Invalid point dimensions detected.", UserWarning)
                    else:
                        warnings.warn(f"Limb {limb}: Indices out of bounds for {num_points} points.", UserWarning)
                except IndexError:
                     warnings.warn(f"Error accessing keypoints for limb {limb}.", UserWarning)
                except Exception as e:
                    warnings.warn(f"Unexpected error drawing limb {limb}: {e}", UserWarning)
            glEnd()
        else:
            # self.statusUpdateRequest.emit("Cannot draw limbs: Limb sequence not loaded.", 3000)
            pass # Don't draw limbs if sequence isn't available


        # --- Draw Keypoints ---
        glPointSize(6.0)
        glColor3f(0.0, 0.5, 1.0) # Light blue points
        glBegin(GL_POINTS)
        try:
            for i, point in enumerate(current_frame_keypoints):
                 # Ensure point is 3D
                if len(point) == 3:
                    glVertex3fv(point)
                else:
                    warnings.warn(f"Keypoint {i}: Invalid point dimensions detected.", UserWarning)
        except Exception as e:
             warnings.warn(f"Unexpected error drawing points: {e}", UserWarning)
        glEnd()

    def mousePressEvent(self, event: QMouseEvent):
        """ Stores the initial mouse position when a button is pressed. """
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        """ Rotates the view based on mouse movement (if left button is pressed). """
        if event.buttons() & Qt.MouseButton.LeftButton:
            dx = event.position().x() - self.last_pos.x()
            dy = event.position().y() - self.last_pos.y()

            # Sensitivity factor - adjust as needed
            sensitivity = 0.4

            # Accumulate rotations
            self.rotation_x += dy * sensitivity
            self.rotation_y += dx * sensitivity

            # Clamp rotation_x to avoid flipping upside down easily
            self.rotation_x = max(-90.0, min(90.0, self.rotation_x))

            self.last_pos = event.position().toPoint()
            self.update() # Trigger repaint

    def wheelEvent(self, event):
        """ Zooms the view using the mouse wheel. """
        delta = event.angleDelta().y()
        # Adjust zoom factor - sensitivity might need tuning
        zoom_sensitivity = 0.005
        self.zoom_factor -= delta * zoom_sensitivity
        # Clamp zoom factor to reasonable limits
        self.zoom_factor = max(1.0, min(self.zoom_factor, 50.0))
        self.update() # Trigger repaint