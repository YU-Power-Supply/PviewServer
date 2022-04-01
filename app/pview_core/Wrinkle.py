import os
import cv2


def wrinkleDetect(file_location):
    img = cv2.imread(file_location)
    img = cv2.resize(img, (256, 256))
    img_blur = cv2.GaussianBlur(img, (0, 0), 1.3)
    img_canny = cv2.Canny(img_blur, 50, 50)

    zero_count = 0
    for i in range(256):
        for j in range(256):
            if img_canny[i][j] == 0:
                zero_count += 1

    imgRatio = 1 - (zero_count/(256*256))
    print(f"주름 : {round(imgRatio,5)}")
    return round(imgRatio, 5)
