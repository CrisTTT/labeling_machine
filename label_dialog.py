from PyQt6.QtWidgets import QDialog, QFormLayout, QLineEdit, QDialogButtonBox

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

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def get_values(self):
        return (self.label_name.text(), self.label_value.text(),
                int(self.start_frame.text()), int(self.end_frame.text()))
