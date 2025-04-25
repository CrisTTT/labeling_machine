# main2.py
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QLabel, QMessageBox
from PyQt6.QtCore import Qt

# Import the refactored interfaces
try:
    from classic_interface import ClassicInterface
except ImportError as e:
    QMessageBox.critical(None, "Import Error", f"Could not import ClassicInterface: {e}\nMake sure classic_interface.py is in the same directory.")
    sys.exit(1)
try:
    from new_interface import NewInterface
except ImportError as e:
     QMessageBox.critical(None, "Import Error", f"Could not import NewInterface: {e}\nMake sure new_interface.py is in the same directory.")
     sys.exit(1)

# Remove pose_labeling_tool import as it's likely redundant


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Pose Labeling Tool Launcher')
        self.setGeometry(200, 200, 350, 200)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint) # Keep launcher on top initially

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel("Choose Labeling Interface")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title_label.font()
        font.setPointSize(14)
        font.setBold(True)
        title_label.setFont(font)
        main_layout.addWidget(title_label)


        self.classic_button = QPushButton('Classic Interface (Interval-Based)', self)
        self.classic_button.setToolTip("Label actions/attributes over time intervals.")
        self.classic_button.setMinimumHeight(40)
        self.classic_button.clicked.connect(self.open_classic_interface)
        main_layout.addWidget(self.classic_button)

        self.new_button = QPushButton('New Interface (Frame-by-Frame)', self)
        self.new_button.setToolTip("Assign a label value to each frame individually.")
        self.new_button.setMinimumHeight(40)
        self.new_button.clicked.connect(self.open_new_interface)
        main_layout.addWidget(self.new_button)

        # Keep track of opened windows to prevent duplicates (optional)
        self.opened_window = None

    def open_classic_interface(self):
        if self.opened_window and self.opened_window.isVisible():
             self.opened_window.activateWindow() # Bring existing window to front
             self.show_status_message("Classic interface is already open.")
             return
        try:
            # Create instance locally, don't store as self.classic_interface
            classic_win = ClassicInterface()
            classic_win.show()
            self.opened_window = classic_win # Store reference
            self.close() # Close the launcher window
        except Exception as e:
             QMessageBox.critical(self, "Error", f"Failed to launch Classic Interface:\n{e}")


    def open_new_interface(self):
        if self.opened_window and self.opened_window.isVisible():
             self.opened_window.activateWindow() # Bring existing window to front
             self.show_status_message("New interface is already open.")
             return
        try:
            # Create instance locally
            new_win = NewInterface()
            new_win.show()
            self.opened_window = new_win # Store reference
            self.close() # Close the launcher window
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch New Interface:\n{e}")

    def show_status_message(self, message):
         # Simple popup for launcher status
         msgBox = QMessageBox(self)
         msgBox.setIcon(QMessageBox.Icon.Information)
         msgBox.setText(message)
         msgBox.setWindowTitle("Info")
         msgBox.setStandardButtons(QMessageBox.StandardButton.Ok)
         msgBox.exec()


def main():
    # Enable High DPI scaling for better visuals on modern displays
    #QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    #QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # Set a style if desired (optional)
    # app.setStyle('Fusion')

    main_window = LauncherWindow()
    main_window.show()

    sys.exit(app.exec())

if __name__ == '__main__':
    # Add error handling for top-level script execution
    try:
        main()
    except Exception as e:
         print(f"Critical Application Error: {e}")
         # Show a final error message if GUI fails early
         try:
             app = QApplication.instance() # Get instance if it exists
             if app is None: app = QApplication(sys.argv) # Create if needed
             QMessageBox.critical(None, "Application Crash", f"A critical error occurred:\n{e}\nThe application will now exit.")
         except:
             pass # Ignore errors during final error reporting
         sys.exit(1)