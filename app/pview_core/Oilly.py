import cv2
import numpy as np

height = 320
width = 320
alpha = 0.8


def pixelDetector(img):
    cnt = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] == 255:
                cnt += 1
    return cnt


def oilly_normal_dry(img):
    return float(pixelDetector(img))/float(height*width)


def contrastControlByHistogram(Img):
    func = (1+alpha) * Img - (alpha * 128)  # 128을 기준으로 명암 맞춰줌
    dst = np.clip(func, 0, 255).astype(np.uint8)
    return dst


def canny(img):
    canny = cv2.Canny(img, 150, 450)  # 2:1 혹은 3:1 의비율을 권함
    return canny


def oilly(file_location):
    oilly = cv2.resize(cv2.imread(file_location, cv2.IMREAD_COLOR), dsize=(width, height))
    temp =  oilly_normal_dry(canny(contrastControlByHistogram(oilly)))
    print("유분 : ", temp*10)
    return temp*10
