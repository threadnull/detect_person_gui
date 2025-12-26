import sys
import os
from PyQt6 import uic, QtGui, QtCore
from PyQt6.QtWidgets import QApplication, QMainWindow

UI_FILE_PATH = "./ui/person.ui"
LOGO_FILE_PATH = "./logo/arkplus.png"

class PersonWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        ui_file_path = UI_FILE_PATH

        if not os.path.exists(ui_file_path):
            print(f"Error: '{ui_file_path}' File not found.")
            sys.exit(1)

        try:
            uic.loadUi(ui_file_path, self)
            self.setWindowTitle("ARKPLUS")            
            self.setup_window_icon()
            self.setup_logo()
            self.adjustSize() 
            
        except Exception as e:
            print(f"UI File Load Error: {e}")
            sys.exit(1)

    def setup_window_icon(self):
        if os.path.exists(LOGO_FILE_PATH):
            self.setWindowIcon(QtGui.QIcon(LOGO_FILE_PATH))
        else:
            print(f"Warning: Not found logo file: {LOGO_FILE_PATH}")

    def setup_logo(self):
        if os.path.exists(LOGO_FILE_PATH):
            pixmap = QtGui.QPixmap(LOGO_FILE_PATH)
            
            if hasattr(self, "label"):
                self.label.setStyleSheet("background-color: white; border: 1px solid #dcdcdc;")
                
                max_width = 180
                max_height = 100
                
                scaled_pixmap = pixmap.scaled(
                    max_width, 
                    max_height, 
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
                
                self.label.setPixmap(scaled_pixmap)
                self.label.setScaledContents(False)
                self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            else:
                print("Warning: Not found label widget in UI file.")

    def resizeEvent(self, event):
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)        
    window = PersonWindow()
    window.show()
    sys.exit(app.exec())