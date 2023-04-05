'''
    분류 : 정밀진단
    목적 : 주름 진단 (서비스용)
'''

import cv2
import numpy as np


# 얼굴 전체 이미지에서 피부결 검출
def globalWrinkleDetect(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # smoothing filter 정의
    smoothing_mask = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    # sharpening filter 정의
    sharpening_mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpening_mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    smoothing_result = cv2.filter2D(imgGray, -1, smoothing_mask)
    sharpening1 = cv2.filter2D(smoothing_result, -1, sharpening_mask1)
    dst = cv2.GaussianBlur(sharpening1, (5, 5), 2)

    sharpening2 = cv2.filter2D(dst, -1, sharpening_mask2)
    dst = cv2.Canny(sharpening2, 180, 220, apertureSize=7)
    
    minLineLength = 100
    maxLineRange = 20
    maxLineGap = 5
    # 피부결 검출
    lines = cv2.HoughLinesP(dst,1,np.pi/180,10,minLineLength,maxLineGap)
    # 이미지에 적용
    inclination = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(imgGray,(x1,y1),(x2,y2),(0,255,0),1)
        if x1 != x2:
            inclination.append((y2-y1)/(x2-x1))
        else:
            inclination.append(maxLineGap)
    
    # 점수 계산
    inclination = sorted(np.array(inclination))
    np.clip(inclination, -1*maxLineRange, maxLineRange)
    size = len(inclination)
    unique, counts = np.unique(inclination, return_counts = True)
    uniqCount = dict(zip(unique, counts))
    # 평균
    average = sum(inclination)/size
    # 중앙값
    median = inclination[round(size/2)]
    # 최빈값
    mode = max(uniqCount,key=uniqCount.get)
    maxError = 10
    error = (abs(mode-average) + abs(mode-median))/10
    score = error # 높을수록 피부결 나쁨
    return round(score,5)

# 부분 이미지에서 주름 검출
def partWrinkleDetect(imgGray):
    #imgGray = cv2.resize(imgGray, (256,256))
    imgBlur = cv2.GaussianBlur(imgGray, (0,0), 1.3)
    imgCanny = cv2.Canny(imgBlur, 50,50)
    rgbPlanes = cv2.split(imgGray)

    resultPlanes = []
    result_norm_planes = []
    for plane in rgbPlanes:
        dilated_img = cv2.dilate(plane, np.ones((5,5), np.uint8)) #RGB 각 채널을 대상으로 커널만큼 팽창시킴
        bg_img = cv2.medianBlur(dilated_img, 21) #블러
        diff_img = 255 - cv2.absdiff(plane, bg_img) #배경제거(그림자로 인식된 부분 제거)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) #표준화
        resultPlanes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(resultPlanes) #병합
    result_norm = cv2.merge(result_norm_planes) #병합

    img_lab = cv2.cvtColor(result, cv2.COLOR_BGR2Lab) # lab convert
    img_l, img_a, img_b = cv2.split(img_lab) # L : 밝기 / A : 초록-빨강 / B : 파랑-노랑

    img_l = img_l - 80 #히스토그램 평행이동
    alpha = 1.5 #스트레칭 비율
    img_ld = np.clip((1+alpha)*img_l - 128*alpha, 0, 255).astype(np.uint8) #히스토그램 스트레칭(명암비 늘리기)
    
    dst = cv2.GaussianBlur(imgGray, (5, 5), 2)
    dst = cv2.Canny(dst, 180, 220, apertureSize=7)

    # 주름확장예측계산
    ret,img_binary = cv2.threshold(img_ld, 210, 255, cv2.THRESH_BINARY_INV) # THRESH_BINARY_INV : 반전된 마스크 이미지
    imgResult = cv2.erode(img_binary, np.ones((3,3), np.uint8))
    
    # 점수 계산
    offset = 0
    scoreCount1 = sum([len(i[i == 0]) for i in imgCanny]) + offset
    scoreCount2 = sum([len(i[i == 0]) for i in imgResult])
    weight = [0.7, 0.3, 5.5]
    score = (1 - ((weight[0]*scoreCount1 + weight[1]*scoreCount2)/(256*256))) * weight[2] # 점수 높을수록 주름 많음
    return int(score * 100) + 10
    #return round(score, 5)

def detect_wrinkle(globalImg, partImg):
    partScore = partWrinkleDetect(partImg)
    globalScore = globalWrinkleDetect(globalImg)
    totalScore = round((partScore*100 + globalScore*10), 5)
    
    if totalScore < 100:
        return totalScore
    else:
        return 97