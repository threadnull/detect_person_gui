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
            print(f"오류: '{ui_file_path}' 파일을 찾을 수 없습니다.")
            sys.exit(1)

        try:
            uic.loadUi(ui_file_path, self)
            self.setWindowTitle("ARKPLUS")
            
            # 변화된 부분 1: 윈도우 아이콘 설정 함수 호출
            self.setup_window_icon()
            
            # 초기 로고(내부 라벨) 설정
            self.setup_logo()
            
            # 변화된 부분 2: 실행 시 창 크기를 UI에 설정된 최소 크기에 맞춤
            self.adjustSize() 
            
        except Exception as e:
            print(f"UI 파일을 로드하는 중 오류가 발생했습니다: {e}")
            sys.exit(1)

    # 변화된 부분 1: 메인 윈도우 아이콘을 설정하는 함수
    def setup_window_icon(self):
        if os.path.exists(LOGO_FILE_PATH):
            # QIcon 객체를 생성하여 윈도우 아이콘으로 설정
            self.setWindowIcon(QtGui.QIcon(LOGO_FILE_PATH))
        else:
            print(f"경고: 아이콘 파일을 찾을 수 없습니다: {LOGO_FILE_PATH}")

    def setup_logo(self):
        if os.path.exists(LOGO_FILE_PATH):
            pixmap = QtGui.QPixmap(LOGO_FILE_PATH)
            
            if hasattr(self, 'label'):
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
                print("경고: UI 파일에서 'label' 위젯을 찾을 수 없습니다.")

    def resizeEvent(self, event):
        super().resizeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 변화된 부분 3: 윈도우 작업표시줄 아이콘이 올바르게 표시되도록 App ID 설정 (Windows 환경용)
    if sys.platform == 'win32':
        import ctypes
        myappid = 'arkplus.opencv_viewer.v1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        
    window = PersonWindow()
    window.show()
    sys.exit(app.exec())