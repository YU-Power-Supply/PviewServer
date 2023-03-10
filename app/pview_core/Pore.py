'''
    분류 : 일반진단
    목적 : 피부 모공 진단 (서비스용)
'''

import cv2
import numpy as np

def detect_pore(img):
    rgb_planes = cv2.split(img)

    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((5,5), np.uint8)) #RGB 각 채널을 대상으로 커널만큼 팽창시킴
        bg_img = cv2.medianBlur(dilated_img, 21) #블러
        diff_img = 255 - cv2.absdiff(plane, bg_img) #배경제거(그림자로 인식된 부분 제거)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) #표준화
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes) #병합

    img_lab = cv2.cvtColor(result_norm, cv2.COLOR_BGR2Lab) # lab convert
    img_l,_, _ = cv2.split(img_lab) # L : 밝기 / A : 초록-빨강 / B : 파랑-노랑

    img_ls = img_l - 70 #히스토그램 앞으로 평행이동(너무 큰 값 살릴려고)
    alpha = 1.5 #스트레칭 비율
    img_lds = np.clip((1+alpha)*img_ls - 128*alpha, 0, 255).astype(np.uint8) #히스토그램 스트레칭(명암비 늘리기)
    _, thr_ld = cv2.threshold(img_lds, 200, 255, cv2.THRESH_BINARY_INV)

    return 100 - round((len(thr_ld[thr_ld>0])*len(thr_ld[thr_ld>0]))/len(img[img>0])/30, 4)
