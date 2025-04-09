import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget
from classic_interface import ClassicInterface
from new_interface import NewInterface

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Pose Labeling Tool')
        self.setGeometry(100, 100, 300, 200)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.classic_button = QPushButton('Classic Interface', self)
        self.classic_button.clicked.connect(self.open_classic_interface)
        main_layout.addWidget(self.classic_button)

        self.new_button = QPushButton('New Interface', self)
        self.new_button.clicked.connect(self.open_new_interface)
        main_layout.addWidget(self.new_button)

    def open_classic_interface(self):
        self.classic_interface = ClassicInterface()
        self.classic_interface.show()

    def open_new_interface(self):
        self.new_interface = NewInterface()
        self.new_interface.show()

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
