# label_dialog.py
from PyQt6.QtWidgets import (QDialog, QFormLayout, QLineEdit,
                             QDialogButtonBox, QMessageBox)
import warnings

# Note: This dialog is currently not used by ClassicInterface or NewInterface.
class LabelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Label")
        self.layout = QFormLayout(self)

        self.label_name = QLineEdit(self)
        self.label_value = QLineEdit(self)
        self.start_frame = QLineEdit(self)
        self.end_frame = QLineEdit(self)

        self.layout.addRow("Label Name:", self.label_name)
        self.layout.addRow("Label Value:", self.label_value)
        self.layout.addRow("Start Frame:", self.start_frame)
        self.layout.addRow("End Frame:", self.end_frame)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self
        )
        self.layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_values(self):
        """
        Retrieves values from the dialog fields.
        Returns tuple (name, value, start_frame, end_frame) or None if invalid.
        """
        name = self.label_name.text().strip()
        value = self.label_value.text().strip()
        start_frame_text = self.start_frame.text().strip()
        end_frame_text = self.end_frame.text().strip()

        if not name:
             QMessageBox.warning(self, "Input Error", "Label Name cannot be empty.")
             return None

        try:
            start_frame = int(start_frame_text)
            end_frame = int(end_frame_text)
            if start_frame > end_frame:
                 QMessageBox.warning(self, "Input Error", "Start Frame cannot be greater than End Frame.")
                 return None
            return (name, value, start_frame, end_frame)
        except ValueError:
            warnings.warn(f"Invalid frame input: Start='{start_frame_text}', End='{end_frame_text}'", UserWarning)
            QMessageBox.warning(self, "Input Error", "Start Frame and End Frame must be valid integers.")
            return None