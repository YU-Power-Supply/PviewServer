import cv2
import numpy as np


def wrinkleDetect(file_location):
    img = cv2.imread(file_location)
    img = cv2.resize(img, (256,256))
    img_blur = cv2.GaussianBlur(img, (0,0), 1.3)
    img_canny = cv2.Canny(img_blur, 50,50)
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((5,5), np.uint8)) #RGB 각 채널을 대상으로 커널만큼 팽창시킴
        bg_img = cv2.medianBlur(dilated_img, 21) #블러
        diff_img = 255 - cv2.absdiff(plane, bg_img) #배경제거(그림자로 인식된 부분 제거)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) #표준화
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes) #병합
    result_norm = cv2.merge(result_norm_planes) #병합

    img_lab = cv2.cvtColor(result, cv2.COLOR_BGR2Lab) # lab convert
    img_l, img_a, img_b = cv2.split(img_lab) # L : 밝기 / A : 초록-빨강 / B : 파랑-노랑

    img_l = img_l - 80 #히스토그램 앞으로 평행이동(너무 큰 값 살릴려고)
    alpha = 1.5 #스트레칭 비율
    img_ld = np.clip((1+alpha)*img_l - 128*alpha, 0, 255).astype(np.uint8) #히스토그램 스트레칭(명암비 늘리기)
    
    hist_l,bins_l = np.histogram(img_l.flatten(),256,[0,256])
    hist_ld,bins_ld = np.histogram(img_ld.flatten(),256,[0,256])
    
    ret,img_binary = cv2.threshold(img_ld, 210, 255, cv2.THRESH_BINARY_INV) # THRESH_BINARY_INV : 반전된 마스크 이미지
    img_result = cv2.erode(img_binary, np.ones((3,3), np.uint8))

    zero_count = sum([len(i[i == 0]) for i in img_result])
    imgRatio = (1 - (zero_count/(256*256))) * 3
    print(f"주름 : {round(imgRatio,5)}")

    return round(imgRatio, 5)

