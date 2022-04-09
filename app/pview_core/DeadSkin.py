import os
import cv2


def deadSkinTest(data_dir, imgName):
    img = cv2.imread(os.path.join(data_dir, imgName))
    img = cv2.resize(img, (256, 256))
    img_blur = cv2.GaussianBlur(img, (0, 0), 1.3)
    img_canny = cv2.Canny(img_blur, 20, 60)

    zero_count = 0
    for i in range(256):
        for j in range(256):
            if img_canny[i][j] == 0:
                zero_count += 1

    imgRatio = 1 - (zero_count/(256*256))
    print(f"각질 : {round(imgRatio*100,5)}")
    return round(imgRatio*100, 5)




"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img # 이미지픽셀을 불러올 모듈


def deadSkin(data_dir, imgName):

    grid = 128

    imgData = load_img(f'{data_dir}/{imgName}', target_size= (grid, grid))

    model = load_model(f"{data_dir}/deadSkin_output_v1.h5")
    testimg = img_to_array(imgData)
    testimg = testimg/255

    h = model.predict(testimg.reshape(1, grid, grid, 3))

    print(f"각질 : {h.argmax()}")
"""
