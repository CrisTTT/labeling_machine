import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QLabel, QMessageBox
from PyQt6.QtCore import Qt
import traceback

try:
    from interface import Interface
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: Could not import Interface: {e}")
    if QApplication.instance():
        QMessageBox.critical(None, "Import Error",
                             f"Could not import Interface: {e}\nMake sure new_interface.py is in the same directory.")
    else:  # Fallback if QApplication itself fails or isn't instantiated
        # Create a temporary app to show the message box
        temp_app = QApplication(sys.argv)
        QMessageBox.critical(None, "Import Error",
                             f"Could not import Interface: {e}\nMake sure new_interface.py is in the same directory.")
    sys.exit(1)
except Exception as e_gen:  # Catch any other exception during import
    print(f"CRITICAL UNEXPECTED IMPORT ERROR (Interface): {e_gen}")
    traceback.print_exc()
    sys.exit(1)


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Pose Labeling Tool Launcher')
        self.setGeometry(200, 200, 350, 150)  # Adjusted height

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel("Pose Labeling Tool")  # Simplified title
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title_label.font()
        font.setPointSize(14)
        font.setBold(True)
        title_label.setFont(font)
        main_layout.addWidget(title_label)

        self.new_button = QPushButton('Open Labeling Interface', self)  # Changed button text
        self.new_button.setToolTip("Open the frame-by-frame 3D pose labeling tool.")
        self.new_button.setMinimumHeight(40)
        self.new_button.clicked.connect(self.open_new_interface)
        main_layout.addWidget(self.new_button)

        self.opened_window = None

    def open_new_interface(self):
        if self.opened_window and self.opened_window.isVisible():
            self.opened_window.activateWindow()
            QMessageBox.information(self, "Info", "Labeling interface is already open.")
            return
        try:
            new_win = Interface()
            new_win.show()
            self.opened_window = new_win
            self.close()  # Close the launcher window
        except Exception as e:
            print(f"ERROR launching New Interface: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Launch Error", f"Failed to launch New Interface:\n{e}")


def main():

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    main_window = LauncherWindow()
    main_window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
    except Exception as e:
        print(f"CRITICAL APPLICATION ERROR in __main__: {e}")
        traceback.print_exc()
        try:
            if not QApplication.instance(): QApplication(sys.argv)
            QMessageBox.critical(None, "Application Crash",
                                 f"A critical error occurred:\n{e}\nThe application will now exit.")
        except:
            pass
        sys.exit(1)
