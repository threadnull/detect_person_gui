# import 로드 순서 변경금지(dll 충돌 예방)
from ultralytics import YOLO
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import Qt
from PyQt6 import uic
import sys, os

# 리소스 절대경로 반환
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # ui 불러오기
        uic.loadUi(resource_path('ui/window.ui'), self)

        # 윈도우 아이콘
        self.setWindowIcon(QIcon(resource_path('logo/arkplus.png')))

        # 각 버튼 콜백
        self.btn_load.clicked.connect(self.load_image)
        self.btn_detect.clicked.connect(self.detect_image)
        self.btn_save.clicked.connect(self.save_image)

        self.image_path = None

        # YOLO11m 모델
        self.model = YOLO(resource_path('model/yolo11m.pt'))

        # 로고
        logo_pixmap = QPixmap(resource_path('logo/arkplus.png'))
        if not logo_pixmap.isNull():
            self.label_logo.setPixmap(logo_pixmap.scaledToWidth(200, Qt.TransformationMode.SmoothTransformation))

    # 이미지 불러오기
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "이미지 열기", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(self.image_path)
            self.label_original.setPixmap(pixmap.scaled(self.label_original.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            # 초기화
            self.label_result.clear()
            self.label_result.setText("결과 이미지")

    # 객체 탐지
    def detect_image(self):
        if self.image_path:
            print(f"Detecting objects in image:{self.image_path}")
            
            results = self.model.predict(self.image_path, classes=[0])
            
            plotted_img = results[0].plot()
            
            # BGR -> RBG
            rgb_img = plotted_img[..., ::-1].copy()

            # qt 이미지로 변환
            h, w, ch = rgb_img.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_img)
            self.label_result.setPixmap(pixmap.scaled(self.label_result.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    # 이미지 저장
    def save_image(self):
        if self.label_result.pixmap():
            file_name, _ = QFileDialog.getSaveFileName(self, "이미지 저장", "result.jpg", "Image Files (*.jpg *.png)")
            if file_name:
                self.label_result.pixmap().save(file_name)

# 엔트리 포인트
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())