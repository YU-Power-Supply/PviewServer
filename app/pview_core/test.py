import cv2
import numpy as np
from random import random

height = 320
width = 320


def run_core(file_location):
    img = cv2.imread(file_location)
    pore_detect = poreDetect(img)
    oilly = oilly_model(img)
    pih = PIH_model(img)
    wrinkle = wrinkleDetect(img)
    skin_tone = skinToneDetect(img)
    dead_skin = round(random(), 2)
    
    return dict(wrinkle=wrinkle, skin_tone=skin_tone,
                pore_detect=pore_detect, dead_skin=dead_skin, oilly=oilly, pih=pih)


#################################### pore ###################################
def poreDetect(img):
    img = cv2.resize(img, (256, 256))

    img_b, img_g, img_r = cv2.split(img)  # B채널만 사용

    m_row = []
    m_column = []
    mrt = []
    for i in img_b:
        sortedList = np.sort(i)
        avr = np.mean(sortedList[int(255*0.1):int(255*0.9)])
        m_row.append(avr)
    rMax = max(m_row)
    m_row = m_row/rMax
    for i in m_row:
        mrt.append([i])

    for i in np.transpose(img_b):
        sortedList = np.sort(i)
        avr = np.mean(sortedList[int(255*0.1):int(255*0.9)])
        m_column.append(avr)
    cMax = max(m_column)
    m_column = [m_column/cMax]

    m_expact = np.dot(mrt, m_column)

    new_gray = img_b/m_expact
    for i in range(256):
        for j in range(256):
            if new_gray[i, j] > 255:
                new_gray[i, j] = 255

    new_gray = new_gray.astype(np.uint8)
    alpha = 0.2  # 얼마나 명암비를 올려줄 것인지에 대한 상수
    new_gray = np.clip((1+alpha)*new_gray - 128*alpha, 0, 255).astype(np.uint8)

    # hist = cv2.calcHist([new_gray], [0], None, [256], [0, 256])

    oneDimArr = sorted(np.ravel(new_gray, order='C'))
    referValue = oneDimArr[int(len(oneDimArr)*0.05)-1]
    # 120값 이미지마다 달라지니까 11.5퍼센트 정도로 찾을수 있는방법찾기
    ret, binaryImg = cv2.threshold(new_gray, referValue, 255, cv2.THRESH_BINARY_INV)

    zero_count = 0
    for i in range(256):
        for j in range(256):
            if binaryImg[i][j] == 0:
                zero_count += 1

    imgRatio = 1 - (zero_count/(256*256))
    print(f"모공 : {round(imgRatio*100,2)}")
    return round(imgRatio*100, 5)

#################################### oilly ###################################


def pixelDetector(img):
    cnt = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 255:
                cnt += 1
    return cnt


def oilly_normal_dry(img):
    skinValue = float(pixelDetector(img))/float(height*width)

    if skinValue > 0.1:
        print(f"유분 : oilly, oilValue : {skinValue*100 : .2f}%")
        return "oilly"
    elif skinValue > 0.05:
        print(f"유분 : normal, oilValue : {skinValue*100 : .2f}%")
        return "normal"
    else:
        print(f"유분 : dry, oilValue : {skinValue*100 : .2f}%")
        return "dry"


def contrastControlByHistogram(Img):
    alpha = 0.8
    func = (1+alpha) * Img - (alpha * 128)  # 128을 기준으로 명암 맞춰줌
    dst = np.clip(func, 0, 255).astype(np.uint8)
    return dst


def canny(img):
    canny = cv2.Canny(img, 150, 450)  # 2:1 혹은 3:1 의비율을 권함
    return canny


def oilly_model(img):
    oilly = cv2.resize(img, (width, height))
    return oilly_normal_dry(canny(contrastControlByHistogram(oilly)))

#################################### pih ###################################


def contrastControlByHistogram(Img):
    alpha = 0.6
    func = (1+alpha) * Img - (alpha * 128)  # 128을 기준으로 명암 맞춰줌
    dst = np.clip(func, 0, 255).astype(np.uint8)
    return dst


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


def PIH_model(img):
    pih_img = cv2.resize(img, (width, height))
    return PIH(contrastControlByHistogram(pih_img))


#################################### wrinke ###################################


def wrinkleDetect(img):
    img = cv2.resize(img, (256, 256))
    img_blur = cv2.GaussianBlur(img, (0, 0), 1.3)
    img_canny = cv2.Canny(img_blur, 50, 50)

    zero_count = 0
    for i in range(256):
        for j in range(256):
            if img_canny[i][j] == 0:
                zero_count += 1

    imgRatio = 1 - (zero_count/(256*256))
    print(f"주름 : {round(imgRatio*100,5)}")
    return round(imgRatio*100, 5)


#################################### skintone ###################################


def skinToneDetect(img):
    imgSize = (256, 256)
    img = cv2.resize(img, imgSize)

    # ----------------------<lab 변환 및 명도 평균값으로 이미지 재생성>--------------------------
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    img_l, img_a, img_b = cv2.split(img_lab)  # L : 밝기 / A : 초록-빨강 / B : 파랑-노랑
    # origin_merge = cv2.merge((img_l, img_a, img_b))

    n_img_l = np.array(img_l)
    avrValue = n_img_l.mean()
    avr_l = np.full(imgSize, avrValue, dtype="uint8")

    # refer = np.full(imgSize, 128, dtype="uint8")
    new_lab = cv2.merge((avr_l, img_a, img_b))
    new_img = cv2.cvtColor(new_lab, cv2.COLOR_Lab2BGR)
# --------------------------<RGB평균값으로 이미지 재생성>-----------------------------
    B_c, G_c, R_c = cv2.split(new_img)
    B_c = np.array(B_c)
    G_c = np.array(G_c)
    R_c = np.array(R_c)

    BAvr = B_c.mean()
    GAvr = G_c.mean()
    RAvr = R_c.mean()

    meanB_C = np.full(imgSize, BAvr, dtype="uint8")
    meanG_C = np.full(imgSize, GAvr, dtype="uint8")
    meanR_C = np.full(imgSize, RAvr, dtype="uint8")

    meanImg = cv2.merge((meanB_C, meanG_C, meanR_C))
    # --------------------------<비교이미지 불러오기 및 색 추출>-----------------------------
    value_img = cv2.imread('app/pview_core/skinToneValue.png')
    value_img = cv2.resize(value_img, imgSize)

    x_ref = 0.1667
    y_ref = 0.25

    grid_1 = [int(imgSize[0]*(x_ref*1)), int(imgSize[1]*(y_ref*1))]
    grid_2 = [int(imgSize[0]*(x_ref*3)), int(imgSize[1]*(y_ref*1))]
    grid_3 = [int(imgSize[0]*(x_ref*5)), int(imgSize[1]*(y_ref*1))]
    grid_4 = [int(imgSize[0]*(x_ref*1)), int(imgSize[1]*(y_ref*3))]
    grid_5 = [int(imgSize[0]*(x_ref*3)), int(imgSize[1]*(y_ref*3))]
    grid_6 = [int(imgSize[0]*(x_ref*5)), int(imgSize[1]*(y_ref*3))]

    skinTone = [value_img[grid_1[0]][grid_1[1]],
                value_img[grid_2[0]][grid_2[1]],
                value_img[grid_3[0]][grid_3[1]],
                value_img[grid_4[0]][grid_4[1]],
                value_img[grid_5[0]][grid_5[1]],
                value_img[grid_6[0]][grid_6[1]]]

    skinTone = np.array(skinTone)

    # --------------------------<피부톤 결과 도출>-----------------------------
    midColor = np.array(meanImg[0][0])
    errorRate = []
    for tone in skinTone:
        errorRate.append(sum(tone-midColor))
    print(f'피부톤 : {(errorRate.index(min(errorRate)) + 1)} type')
    # cv2.waitKey(0)
    return (errorRate.index(min(errorRate)) + 1)
