
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Person Detector")
        self.setGeometry(100, 100, 1000, 600)

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Image display area
        image_layout = QHBoxLayout()
        self.label_original = QLabel("Original Image")
        self.label_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_original.setFixedSize(480, 480)
        self.label_original.setStyleSheet("border: 1px solid black;")
        image_layout.addWidget(self.label_original)

        self.label_result = QLabel("Result Image")
        self.label_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_result.setFixedSize(480, 480)
        self.label_result.setStyleSheet("border: 1px solid black;")
        image_layout.addWidget(self.label_result)
        main_layout.addLayout(image_layout)

        # Buttons layout
        button_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Image")
        self.btn_detect = QPushButton("Detect")
        self.btn_save = QPushButton("Save Image")

        button_layout.addWidget(self.btn_load)
        button_layout.addWidget(self.btn_detect)
        button_layout.addWidget(self.btn_save)
        main_layout.addLayout(button_layout)

        # Connect buttons to functions
        self.btn_load.clicked.connect(self.load_image)
        self.btn_detect.clicked.connect(self.detect_image)
        self.btn_save.clicked.connect(self.save_image)

        self.image_path = None

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(self.image_path)
            self.label_original.setPixmap(pixmap.scaled(self.label_original.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def detect_image(self):
        # Placeholder for YOLO detection
        if self.image_path:
            print(f"Detecting persons in {self.image_path}")
            # Here we will add the YOLO detection logic
            # For now, let's just copy the image to the result label
            pixmap = QPixmap(self.image_path)
            self.label_result.setPixmap(pixmap.scaled(self.label_result.size(), Qt.AspectRatioMode.KeepAspectRatio))


    def save_image(self):
        # Placeholder for saving the result image
        if self.label_result.pixmap():
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "result.jpg", "Image Files (*.jpg *.png)")
            if file_name:
                self.label_result.pixmap().save(file_name)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
