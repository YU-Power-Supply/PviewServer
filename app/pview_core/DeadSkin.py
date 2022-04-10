import cv2
import numpy as np

height = 280
width = 280
alpha = 0.6

def pixelDetector(img):
    cnt = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] >0:
                cnt += 1
    return cnt/(height*width)

def fillterCustom1(img):
    gx_k = np.array([[-3,0,3], [-10,0,10],[-3,0,3]])
    gy_k = np.array([[-3,-10,-3],[0,0,0], [3,10,3]])
    edge_gx = cv2.filter2D(img, -1, gx_k)
    edge_gy = cv2.filter2D(img, -1, gy_k)
    scharrx = cv2.Scharr(img, -1, 1, 0)
    scharry = cv2.Scharr(img, -1, 0, 1)

    # merged1 = np.hstack((img, edge_gx, edge_gy))
    # merged2 = np.hstack((img, scharrx, scharry))
    # merged = np.vstack((merged1, merged2))
    return (edge_gy+edge_gx)/320

def fillterCustom2(img):
    gx_k = np.array([[-3,3,3], [-10,3,10],[-3,3,3]])
    gy_k = np.array([[-3,-10,-3],[3,3,3], [3,10,3]])
    edge_gx = cv2.filter2D(img, -1, gx_k)
    edge_gy = cv2.filter2D(img, -1, gy_k)

    board = np.zeros_like(img)
    board[(edge_gy+edge_gx)/320 > 0] = 255

    return board

def deadSkin_model_dataPrepocessing(img):
    image_b, image_g, image_r = cv2.split(img)
    avrColor = (image_b.sum()/(width*height), image_g.sum()/(width*height), image_r.sum()/(width*height))

    # img = cv2.Canny(img, 40, 70)

    image_b = np.clip((1+alpha) * image_b - (alpha * avrColor[0]), 0, 255).astype(np.uint8)
    image_g = np.clip((1+alpha) * image_g - (alpha * avrColor[1]), 0, 255).astype(np.uint8)
    image_r = np.clip((1+alpha) * image_r - (alpha * avrColor[2]), 0, 255).astype(np.uint8)
    
    img = cv2.merge((image_b, image_g, image_r))
    img = fillterCustom1(img)
    image_b, image_g, image_r = cv2.split(img)
    img = (image_r+image_g+image_b)/3
    return img

def deadSkin_model_dataPrepocessing_custom(img):
    image_b, image_g, image_r = cv2.split(img)
    avrColor = (image_b.sum()/(width*height), image_g.sum()/(width*height), image_r.sum()/(width*height))

    image_b = np.clip((1+alpha) * image_b - (alpha * avrColor[0]), 0, 255).astype(np.uint8)
    image_g = np.clip((1+alpha) * image_g - (alpha * avrColor[1]), 0, 255).astype(np.uint8)
    image_r = np.clip((1+alpha) * image_r - (alpha * avrColor[2]), 0, 255).astype(np.uint8)
    
    img = cv2.merge((image_b, image_g, image_r))
    img = fillterCustom1(img)
    img = fillterCustom2(img)
    image_b, image_g, image_r = cv2.split(img)
    img = (image_r+image_g+image_b)/3
    
    return img


###     아래 코드 실행할 것     ###
'''
    deadskin_detector(src) -> value 0 .. 1 (dtype = float)
    Ex) deadskin_detector("datas/deadskin01.jpg") -> 0.87
'''
def deadSkinDetect(file_location):
    img = cv2.resize(cv2.imread(file_location), dsize = (width, height))
    img_deadskin_detected = deadSkin_model_dataPrepocessing(img)
    img_deadskin_detected2 = deadSkin_model_dataPrepocessing_custom(img)
    
    temp = round(1-((100*(np.clip(pixelDetector(img_deadskin_detected- img_deadskin_detected2), 0.1, 0.16)-0.1))/6), 5)
    print("각질 : " , temp)
    return temp
