from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt6.QtCore import QPoint

class OpenGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rotation_x = 0
        self.rotation_y = 0
        self.last_pos = QPoint()
        self.frame_index = 0
        self.keypoints = None
        self.limbSeq = None

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        gluPerspective(45, (600 / 600), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)

    def paintGL(self):
        try:
            if self.keypoints is None or self.limbSeq is None:
                return

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glPointSize(5.0)
            glColor3f(0.0, 0.0, 1.0)

            glLoadIdentity()
            gluPerspective(45, (600 / 600), 0.1, 50.0)
            glTranslatef(0.0, 0.0, -5)
            glRotatef(self.rotation_x, 1, 0, 0)
            glRotatef(self.rotation_y, 0, 1, 0)

            self.draw_axes()
            self.draw_pose()
        except Exception as e:
            print(f"Error in paintGL: {e}")

    def draw_axes(self):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.0)
        glEnd()

    def draw_pose(self):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(0.0, 0.0, 0.0)
        for limb in self.limbSeq:
            start_point = self.keypoints[self.frame_index][limb[0]]
            end_point = self.keypoints[self.frame_index][limb[1]]
            glVertex3f(start_point[0], start_point[1], start_point[2])
            glVertex3f(end_point[0], end_point[1], end_point[2])
        glEnd()

        glBegin(GL_POINTS)
        for point in self.keypoints[self.frame_index]:
            glVertex3f(point[0], point[1], point[2])
        glEnd()

    def mousePressEvent(self, event):
        self.last_pos = event.position().toPoint()

    def mouseMoveEvent(self, event):
        dx = event.position().x() - self.last_pos.x()
        dy = event.position().y() - self.last_pos.y()

        self.rotation_x += dy
        self.rotation_y += dx

        self.last_pos = event.position().toPoint()
        self.update()
