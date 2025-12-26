import cv2
import numpy as np
from rknnlite.api import RKNNLite

# 하이퍼 파라미터
MODEL_PATH = "./model/yolo11n_rk3588.rknn"
CAMERA_INDEX = 0
INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45

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

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
def post_process(outputs, conf_threshold, nms_threshold):
    # YOLOv11[1, 84, 8400](class: 80, box: 4) -> (84, 8400)
    predictions = np.squeeze(outputs[0]).T

    # 점수, 클래스 추출
    scores = predictions[:, 4:]
    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)

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
        for i in indices.flatten():
            person_boxs.append(boxes[i])
            person_scores.append(confidences[i])
            person_cls_ids.append(class_ids[i])

    return person_boxs, person_scores, person_cls_ids

# 추론
def detect_object():
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

    # 타이머
    tm = cv2.TickMeter()

    if not cap.isOpened():
        print("Camera not found")
        return

    # 카메라 세팅(MJGP 1280x480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # print(f"{int(width)}x{int(height)}")

    while cap.isOpened():
        tm.start()

        ret, frame = cap.read()
        if not ret:
            break

        # 왼쪽 카메라만 사용
        h, w = frame.shape[:2]
        half_w = w // 2
        left_camera = frame[0:h, half_w:w]
        frame = left_camera

        # 전처리
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, INPUT_SIZE)
        img = np.expand_dims(img, axis=0)

        # 추론
        outputs = rknn_lite.inference(inputs=[img])

        # 후처리
        boxes, scores, class_ids = post_process(outputs, CONF_THRESHOLD, NMS_THRESHOLD)

        tm.stop()
        fps = tm.getFPS()

        # 화면 출력
        scale_x, scale_y = frame.shape[1] / INPUT_SIZE[0], frame.shape[0] / INPUT_SIZE[1]
        for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h = box
            x1, y1 = int(x * scale_x), int(y * scale_y)
            x2, y2 = int((x + w) * scale_x), int((y + h) * scale_y)

            color = COLORS[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{CLASSES[class_id]} {score:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"FPS: {round(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("OrangePi 5 Plus with YOLO11n model Person Detection", frame)

        tm.reset()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    rknn_lite.release()

if __name__ == "__main__":
    detect_object()