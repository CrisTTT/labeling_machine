import sys
from PyQt6.QtWidgets import QApplication
from pose_labeling_tool import PoseLabelingTool

def main():
    app = QApplication(sys.argv)
    tool = PoseLabelingTool()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
