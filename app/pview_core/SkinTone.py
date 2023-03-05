import numpy as np
import cv2


def averaging_skintone(img, side_length):

    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #HSV 영역으로 색변환
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) # YCrCb 영역으로 색변환

    # display_image(img_HSV, "HSV")
    # display_image(img_YCrCb, "YCrCb")

    # aggregate skin pixels
    blue = []
    green = []
    red = []

    # 모든 픽셀에 대해 피부라고 판단되는 영역만 검출
    for i in range (side_length):
        for j in range (side_length):
            if((img_HSV.item(i, j, 0) <= 170) and (140 <= img_YCrCb.item(i, j, 1) <= 170) and (90 <= img_YCrCb.item(i, j, 2) <= 120)):
                blue.append(img[i, j].item(0))
                green.append(img[i, j].item(1))
                red.append(img[i, j].item(2))
            else:
                img[i, j] = [0, 0, 0]
    
    # display_image(img_face_only, "final segmentation")

    # determine mean skin tone estimate
    skin_tone_estimate_BGR = [np.mean(blue), np.mean(green), np.mean(red)]
    return skin_tone_estimate_BGR


def skinToneDetect(image, side_length = 256, pixel_range = 16): # 원래값은 540, 40 이었음
    
    # 피부톤 컬러차트
    color_chart = [  (0,0,0),         # 0. 점수 없음
                (212,230,248),  # 1. Ecru (가장 밝음)
                (195,220,245),  # 2. Beige
                (187,214,243),  # 3. Moccasin
                (161,198,239),  # 4. Fawn
                (144,188,236),  # 5. Tan
                (119,172,231),  # 6. Wren
                (93,157,227),   # 7. Cinnamon
                (76,146,224),   # 8. Tawny
                (59,136,221),   # 9. Nutmeg
                (35,116,205),   # 10. Copper
                (26,87,154),    # 11. Woodland
                (16,53,94)] 

    # 이미지 입력 #
    image = cv2.resize(image, (side_length, side_length))
    global_RGB = averaging_skintone(image, side_length)
    global_error = [sum(np.abs(np.array(global_RGB) - np.array(tone))) for tone in color_chart]
    global_value = global_error.index(min(global_error))

    grid_value = []
    division = side_length//pixel_range
    for mode in (0,1):
        for i, start_value in enumerate(range(0,side_length, division), start=1):
            if mode == 0:
                split_img = image[start_value:i*division, :]
            elif mode == 1:
                split_img = image[:, start_value:i*division]

            img_b, img_g, img_r = cv2.split(split_img)
            
            if len(img_b[img_b>0]) == 0:
                filtered_img_b = [0]
            else:
                filtered_img_b = img_b[img_b>0]
            if len(img_g[img_g>0]) == 0:
                filtered_img_g = [0]
            else:
                filtered_img_g = img_g[img_g>0]
            if len(img_r[img_r>0]) == 0:
                filtered_img_r = [0]
            else:
                filtered_img_r = img_r[img_r>0]

            rgb_value = [np.mean(filtered_img_b), np.mean(filtered_img_g), np.mean(filtered_img_r)]
            
            errorRate = []
            for tone in color_chart:
                errorRate.append(sum(np.abs(np.array(rgb_value) - np.array(tone))))
            partValue = errorRate.index(min(errorRate))
            grid_value.append(partValue)

    gridWeight = 0.8
    return 100 - round(sum(np.array(grid_value)*0.29761) * gridWeight + global_value * (1-gridWeight), 1)
    
