
# 필요한 라이브러리들을 임포트합니다.
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import uic # Qt Designer로 만든 UI 파일을 로드하기 위한 uic 모듈

# 메인 윈도우 클래스를 정의합니다.
class MainWindow(QMainWindow):
    # 생성자 함수
    def __init__(self):
        super().__init__()
        # uic.loadUi를 사용하여 Qt Designer에서 만든 'detect.ui' 파일을 로드합니다.
        uic.loadUi('detect.ui', self)

        # UI에 있는 버튼들을 클릭했을 때 호출될 함수들을 연결합니다.
        self.btn_load.clicked.connect(self.load_image)
        self.btn_detect.clicked.connect(self.detect_image)
        self.btn_save.clicked.connect(self.save_image)

        self.image_path = None # 현재 로드된 이미지의 경로를 저장할 변수

    # '이미지 로드' 버튼 클릭 시 실행될 함수
    def load_image(self):
        # 파일 대화상자를 열어 이미지 파일을 선택합니다.
        file_name, _ = QFileDialog.getOpenFileName(self, "이미지 열기", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(self.image_path)
            # 원본 이미지 레이블에 이미지를 표시합니다.
            # 이미지를 레이블 크기에 맞게 조절하고, 비율을 유지합니다.
            self.label_original.setPixmap(pixmap.scaled(self.label_original.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    # '객체 탐지' 버튼 클릭 시 실행될 함수
    def detect_image(self):
        # YOLO 객체 탐지 로직을 위한 자리표시자입니다.
        if self.image_path:
            print(f"{self.image_path} 에서 사람을 탐지합니다.")
            # 이곳에 YOLO 탐지 로직이 추가될 것입니다.
            # 지금은 원본 이미지를 결과 레이블에 복사하는 것으로 대체합니다.
            pixmap = QPixmap(self.image_path)
            # 결과 이미지 레이블에 이미지를 표시합니다.
            self.label_result.setPixmap(pixmap.scaled(self.label_result.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    # '이미지 저장' 버튼 클릭 시 실행될 함수
    def save_image(self):
        # 결과 이미지 저장을 위한 자리표시자입니다.
        if self.label_result.pixmap():
            # 파일 저장 대화상자를 열어 저장할 경로를 선택합니다.
            file_name, _ = QFileDialog.getSaveFileName(self, "이미지 저장", "result.jpg", "Image Files (*.jpg *.png)")
            if file_name:
                # 결과 레이블의 이미지를 파일로 저장합니다.
                self.label_result.pixmap().save(file_name)

# 이 스크립트가 직접 실행될 때 아래 코드를 실행합니다.
if __name__ == "__main__":
    app = QApplication(sys.argv) # QApplication 객체를 생성합니다.
    window = MainWindow() # MainWindow 객체를 생성합니다.
    window.show() # 윈도우를 화면에 보여줍니다.
    sys.exit(app.exec()) # 이벤트 루프를 시작하고, 프로그램이 종료될 때까지 유지합니다.
