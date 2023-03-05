import cv2
import numpy as np


def wrinkleDetect(img, img_size = 560):
    img_gray = cv2.resize(img, (img_size,img_size))
    img_blur = cv2.GaussianBlur(img_gray, (0,0), 1.3)
    img_canny = cv2.Canny(img_blur, 50,50)
    rgb_planes = cv2.split(img)

    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((5,5), np.uint8)) #RGB 각 채널을 대상으로 커널만큼 팽창시킴
        bg_img = cv2.medianBlur(dilated_img, 21) #블러
        diff_img = 255 - cv2.absdiff(plane, bg_img) #배경제거(그림자로 인식된 부분 제거)
        result_planes.append(diff_img)

    result = cv2.merge(result_planes) #병합

    img_lab = cv2.cvtColor(result, cv2.COLOR_BGR2Lab) # lab convert
    img_l, _, _ = cv2.split(img_lab) # L : 밝기 / A : 초록-빨강 / B : 파랑-노랑

    img_l = img_l - 80 #히스토그램 평행이동
    alpha = 1.5 #스트레칭 비율
    img_ld = np.clip((1+alpha)*img_l - 128*alpha, 0, 255).astype(np.uint8) #히스토그램 스트레칭(명암비 늘리기)
    
    dst = cv2.GaussianBlur(img_gray, (5, 5), 2)
    dst = cv2.Canny(dst, 180, 220, apertureSize=7)
    
    # 주름확장예측계산
    _, img_binary = cv2.threshold(img_ld, 210, 255, cv2.THRESH_BINARY_INV) # THRESH_BINARY_INV : 반전된 마스크 이미지
    img_result = cv2.erode(img_binary, np.ones((3,3), np.uint8))
    
    # 점수 계산
    offset = 10
    score_count1 = sum([len(i[i == 0]) for i in img_canny]) + offset
    score_count2 = sum([len(i[i == 0]) for i in img_result])
    weight = [0.7, 0.3, 1.5]
    score = (((weight[0]*score_count1 + weight[1]*score_count2)/(2*256*256))) * weight[2] # 점수 높을수록 주름 많음
    return 100 - round(score, 4)