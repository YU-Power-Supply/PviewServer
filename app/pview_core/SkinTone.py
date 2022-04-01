import numpy as np
import cv2


def skinToneDetect(file_location):

    imgSize = (256, 256)

    img = cv2.imread(file_location)
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

    # cv2.imshow('origin', img) #원본이미지
    # cv2.imshow('lab', img_lab) #lab공간으로 변환된 이미지
    # cv2.imshow('labcvt', img_lab_brg) #lab으로 갔다가 다시 BGR로 합친 이미지

    # cv2.imshow('lc', l_c)
    # cv2.imshow('ac', a_c)
    # cv2.imshow('bc', b_c)
    # cv2.imshow('new', new_img)
    # cv2.imshow('meanImg', meanImg)
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
