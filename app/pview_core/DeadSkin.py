'''
    분류 : 정밀진단
    목적 : 피부 각질 검출 (서비스용)
'''

import cv2
import numpy as np
import copy
from app.pview_core.utils import pixelDetector, saturate_contrastA


## HyperParam
height, width = 256, 256
alpha = 0.6

def fillterCustom1(img):
    gx_k = np.array([[-3,0,3], [-10,0,10],[-3,0,3]])
    gy_k = np.array([[-3,-10,-3],[0,0,0], [3,10,3]])
    edge_gx = cv2.filter2D(img, -1, gx_k)
    edge_gy = cv2.filter2D(img, -1, gy_k)

    return (edge_gy+edge_gx)/320

def fillterCustom2(img):
    gx_k = np.array([[-3,3,3], [-10,3,10],[-3,3,3]])
    gy_k = np.array([[-3,-10,-3],[3,3,3], [3,10,3]])
    edge_gx = cv2.filter2D(img, -1, gx_k)
    edge_gy = cv2.filter2D(img, -1, gy_k)

    board = np.zeros_like(img)
    board[(edge_gy+edge_gx)/320 > 0] = 255

    return board

def fillterCustom3(img):
    gy_k = np.array([[-3,-3,-3],[3,3,3], [3,3,3]])
    edge_gy = cv2.filter2D(img, -1, gy_k)

    return edge_gy

def fillterCustom4(img):
    gy_k = np.array([[-3,-10,-3],[0,0,0], [3,10,3]])
    edge_gy = cv2.filter2D(img, -1, gy_k)

    return edge_gy


def deadSkin_model_dataPrepocessing_type_fst_layer(img):
    image_b, image_g, image_r = cv2.split(img)
    avrColor = (image_b.sum()/(width*height), image_g.sum()/(width*height), image_r.sum()/(width*height))

    image_b = np.clip((1+alpha) * image_b - (alpha * avrColor[0]), 0, 255).astype(np.uint8)
    image_g = np.clip((1+alpha) * image_g - (alpha * avrColor[1]), 0, 255).astype(np.uint8)
    image_r = np.clip((1+alpha) * image_r - (alpha * avrColor[2]), 0, 255).astype(np.uint8)
    
    img = cv2.merge((image_b, image_g, image_r))
    img = fillterCustom1(img)
    image_b, image_g, image_r = cv2.split(img)
    img = (image_r+image_g+image_b)/3

    return img

def deadSkin_model_dataPrepocessing_type_scd_layer(img):
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

def deadSkin_model_dataPrepocessing_type_thd_layer(img):
    image_b, image_g, image_r = cv2.split(img)
    avrColor = (image_b.sum()/(width*height), image_g.sum()/(width*height), image_r.sum()/(width*height))

    image_b = np.clip((1+alpha) * image_b - (alpha * avrColor[0]), 0, 255).astype(np.uint8)
    image_g = np.clip((1+alpha) * image_g - (alpha * avrColor[1]), 0, 255).astype(np.uint8)
    image_r = np.clip((1+alpha) * image_r - (alpha * avrColor[2]), 0, 255).astype(np.uint8)
    
    img = cv2.merge((image_b, image_g, image_r))
    img = fillterCustom3(img)
    image_b, image_g, image_r = cv2.split(img)
    img = (image_r+image_g+image_b)/3
    
    return img

'''
    deadskin_detector(src) -> value 0 .. 1 (dtype = float)
    Ex) deadskin_detector("datas/deadskin01.jpg") -> 0.87
'''

def detect_deadskin(img, model_path):
    #img = cv2.resize(img, dsize = (width, height))
    img_deadskin_detected = deadSkin_model_dataPrepocessing_type_fst_layer(img)
    img_deadskin_detected2 = deadSkin_model_dataPrepocessing_type_scd_layer(img)
    img_deadskin_detected3 = deadSkin_model_dataPrepocessing_type_thd_layer(img)

    result_img1 = img_deadskin_detected - img_deadskin_detected2
    result_img2 = img_deadskin_detected3
    result_img3 = img_deadskin_detected
    result_img4 = img_deadskin_detected2

    ## ITA(멜라닌지수)를 이용한 추출
    img_cie_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    cie_l, _, cie_b = cv2.split(img_cie_lab)
    ita = ((np.arctan((cie_l-50)/cie_b))*180/np.pi).astype(np.uint8)

    ita_strong = copy.deepcopy(ita)
    ita_strong[ita_strong>np.mean(ita)] = 255

    ita_main = saturate_contrastA(ita, 50, 1.8)
    remove_ita = saturate_contrastA(ita, 50, 2.5)

    # remove 노이즈제거
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # 구조화 요소커널 생성
    remove_noise_ita = cv2.morphologyEx(cv2.morphologyEx(remove_ita, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    ita_main[remove_noise_ita>0] = 0
    result = (pixelDetector(ita_main)/pixelDetector(result_img1) + (pixelDetector(result_img2)+pixelDetector(result_img3)+pixelDetector(result_img4))/(pixelDetector(result_img1)*3))*10
    result = (100-result)*3
    result = np.clip(result, 0, 100).astype(np.uint8)
    
    # moisture
    result = 100 - result
    standard_value, boundary_value = 43, 20
    result = - (result - standard_value)  + standard_value 
    result = np.clip(result, 0, 100).astype(np.uint8)
    result = np.clip(result + boundary_value, 0, 100).astype(np.uint8)

    return result