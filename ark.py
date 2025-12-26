import sys
import os
import cv2
import numpy as np
from rknnlite.api import RKNNLite
from PyQt6 import uic, QtGui
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QThread, pyqtSignal, Qt

# 하이퍼 파라미터
MODEL_PATH = "./model/yolo11n_rk3588.rknn"
CAMERA_INDEX = 0
INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

UI_FILE_PATH = "./ui/person.ui"
LOGO_FILE_PATH = "./logo/arkplus.png"
PALETTE = [(0,255,0),]

# YOLO기본 클래스
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

COLORS = np.array(PALETTE, dtype=np.uint8)
if COLORS.shape[0] < len(CLASSES):
    repeats = int(np.ceil(len(CLASSES) / COLORS.shape[0]))
    COLORS = np.tile(COLORS, (repeats, 1))[:len(CLASSES)]

def post_process(outputs, conf_threshold, nms_threshold):
    # YOLOv11[1, 84, 8400](class: 80, box: 4) -> (84, 8400)
    predictions = np.squeeze(outputs[0]).T

    scores_raw = predictions[:, 4:]

    # 점수, 클래스 추출
    scores = predictions[:, 4:]
    max_scores = np.max(scores_raw, axis=1)
    class_ids = np.argmax(scores_raw, axis=1)

    # 필터
    mask = (max_scores > conf_threshold) & (class_ids == 0)

    preds = predictions[mask]
    scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(preds) == 0:
        return [], [], []
    
    # 필터링된 박스좌표(cx, cy, w, h -> x, y, w, h)
    w = preds[:, 2]
    h = preds[:, 3]
    x = preds[:, 0] - w/2
    y = preds[:, 1] - h/2

    # NMSbox
    boxes = np.stack((x, y, w, h), axis=1).tolist()
    confidences = scores.tolist()

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    person_boxs = []
    person_scores = []
    person_cls_ids = []

    if len(indices) > 0:
        for i in indices.ravel():
            person_boxs.append(boxes[i])
            person_scores.append(confidences[i])
            person_cls_ids.append(class_ids[i])

    return person_boxs, person_scores, person_cls_ids

# 추론
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        rknn_lite = RKNNLite()

        print(f"Loading model: {MODEL_PATH}")
        if rknn_lite.load_rknn(MODEL_PATH) != 0:
            print("Model load fail")
            return

        print("Init runtime")
        if rknn_lite.init_runtime() != 0:
            print("Init runtime fail")
            return

        cap = cv2.VideoCapture(CAMERA_INDEX)
        tm = cv2.TickMeter()

        if not cap.isOpened():
            print("Camera not found")
            return

        # 카메라 세팅(MJGP 1280x480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self._run_flag and cap.isOpened():
            tm.start()
            ret, frame = cap.read()
            if not ret:
                break

            # 왼쪽 카메라만 사용
            h, w = frame.shape[:2]
            frame = frame[0:h, w//2:w].copy()

            # 전처리
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_input = cv2.resize(img, INPUT_SIZE)
            img_input = np.expand_dims(img_input, axis=0)

            # 추론
            outputs = rknn_lite.inference(inputs=[img_input])

            # 후처리
            boxes, scores, class_ids = post_process(outputs, CONF_THRESHOLD, NMS_THRESHOLD)

            tm.stop()
            fps = tm.getFPS()

            # 박스 그리기
            scale_x, scale_y = frame.shape[1] / INPUT_SIZE[0], frame.shape[0] / INPUT_SIZE[1]
            for box, score, class_id in zip(boxes, scores, class_ids):
                x, y, w, h = box
                x1, y1 = int(x * scale_x), int(y * scale_y)
                x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)

                color = tuple(int(color) for color in COLORS[class_id])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{CLASSES[class_id]} {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(frame, f"FPS: {round(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            self.change_pixmap_signal.emit(frame)
            tm.reset()

        cap.release()
        rknn_lite.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# GUI 출력
class PersonWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        if not os.path.exists(UI_FILE_PATH):
            print(f"Error: '{UI_FILE_PATH}' File not found.")
            sys.exit(1)

        try:
            uic.loadUi(UI_FILE_PATH, self)
            self.setWindowTitle("ARKPLUS")

            # 꽉찬 화면
            self.showMaximized()
            
            if os.path.exists(LOGO_FILE_PATH):
                # 아이콘
                self.setWindowIcon(QtGui.QIcon(LOGO_FILE_PATH))
                
                # 로고
                logo_pixmap = QtGui.QPixmap(LOGO_FILE_PATH)
                if hasattr(self, "label"):
                    self.label.setStyleSheet("background-color: white; border: 1px solid #dcdcdc;")
                    self.label.setPixmap(logo_pixmap.scaled(
                        self.label.width(), self.label.height(), 
                        Qt.AspectRatioMode.KeepAspectRatio, 
                        Qt.TransformationMode.SmoothTransformation
                    ))
                    self.label.setText("")
            
            self.thread = VideoThread()

            # 영상
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()
            
        except Exception as e:
            print(f"UI File Load Error: {e}")
            sys.exit(1)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        if hasattr(self, "video_display"):
            pixmap = qt_img.scaled(self.video_display.size(), 
                                   Qt.AspectRatioMode.KeepAspectRatio, 
                                   Qt.TransformationMode.SmoothTransformation)
            self.video_display.setPixmap(pixmap)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        return QtGui.QPixmap.fromImage(convert_to_Qt_format)

if __name__ == "__main__":
    app = QApplication(sys.argv)        
    window = PersonWindow()
    window.show()
    sys.exit(app.exec())