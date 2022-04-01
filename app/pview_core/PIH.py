import cv2
import numpy as np

height = 320
width = 320
alpha = 0.6

def contrastControlByHistogram(Img, alpha):
    func = (1+alpha) * Img - (alpha * 128)  # 128을 기준으로 명암 맞춰줌
    dst = np.clip(func, 0, 255).astype(np.uint8)
    return dst

# Canny Edge
# cv2.Canny(image, threshold1, threshold2, edge = None, apertureSize = None, L2gradient = None)
# Gausian Blur
# cv2.GaussianBlur(image, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
# [sigmaX, sigmaY : x, y 편향]
# [ksize : 가우시안 커널 크기, (0, 0)을 지정하면 sigma 값에 의해 자동 결정됨]
# [borderType : 가장자리 픽셀 확장 방식]


def PIH(img):

    # 특징점 알고리즘 객체 생성 (KAZE, AKAZE, ORB 등)
    feature = cv2.KAZE_create(threshold=0.0002)  # 방향 성분은 표현이 안됌
    # feature = cv2.AKAZE_create() # 카제를 빠르게, accelateKaze, 방향선분 표현
    # feature = cv2.ORB_create() # 가장 빠르지만 성능이 떨어짐

    # 특징점 검출
    kp1 = feature.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # kp2 = feature.detect(src2)

    # 검출된 특징점 갯수 파악
    print(f"색소침착 : {len(kp1)}")

    # 검출된 특징점 출력 영상 생성
    img = cv2.drawKeypoints(img, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return len(kp1)  # img


def PIH_model(file_location):
    pih_img = (cv2.resize(cv2.imread(file_location, cv2.IMREAD_COLOR), dsize=(width, height)))
    return PIH(contrastControlByHistogram(pih_img, 0.6))