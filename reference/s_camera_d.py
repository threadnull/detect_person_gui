import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 1. StereoBM 설정
num_disp = 96 # 시차 범위 (16의 배수)
block_size = 15 # 블록 크기 (홀수)
left_matcher = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)

# [추가됨] 2. WLS 필터 설정을 위한 Right Matcher 생성
# WLS 필터는 왼쪽과 오른쪽에서 각각 계산된 시차 지도를 비교하여 노이즈를 찾습니다.
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# [추가됨] 3. WLS 필터 객체 생성 및 파라미터 설정
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000.0)    # 필터의 강도 (값이 클수록 부드러워짐, 보통 8000)
wls_filter.setSigmaColor(1.5)   # 테두리를 얼마나 잘 보존할지 (보통 1.0~2.0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    half_width = w // 2
    left_camera = frame[0:h, 0:half_width]
    right_camera = frame[0:h, half_width:w]

    # WLS 필터 성능 향상을 위해 그레이스케일 변환
    left_gray = cv2.cvtColor(left_camera, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_camera, cv2.COLOR_BGR2GRAY)

    # [변경됨] 4. 양방향 Disparity 계산
    disparity_left = left_matcher.compute(left_gray, right_gray)
    disparity_right = right_matcher.compute(right_gray, left_gray)

    # [추가됨] 5. WLS 필터 적용 (최종 노이즈 제거 및 구멍 메우기)
    # 원본 왼쪽 이미지를 가이드로 사용하여 테두리를 선명하게 유지합니다.
    filtered_disp = wls_filter.filter(disparity_left, left_gray, disparity_map_right=disparity_right)

    # [변경됨] 6. 시각화 처리
    # 필터 결과물은 16비트이므로 정규화가 필요합니다.
    norm_disparity = cv2.normalize(src=filtered_disp, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    depth_map = cv2.applyColorMap(norm_disparity, cv2.COLORMAP_JET)

    combined_frame = cv2.hconcat([left_camera, depth_map])
    cv2.imshow("View", combined_frame)
    
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()