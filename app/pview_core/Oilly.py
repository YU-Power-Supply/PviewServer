'''
    분류 : 일반진단
    목적 : 피부 유분 진단 (서비스용)
'''

import numpy as np
import cv2
from app.pview_core.utils import saturate_contrastB, saturate_contrastA

def oil_detector(img, model):
    # 명도 평준화 이미지
    img = saturate_contrastB(img, 200)
    
    # 명도 양극화 이미지
    img = saturate_contrastA(img, 128, 0.5)

    # 벡터이미지
    #output = img_polarization
    for _ in range(5): 
        img = saturate_contrastB(img, 140)
    for _ in range(2):
        img = saturate_contrastA(img, 50, 0.5)
    for _ in range(3): 
        img = saturate_contrastB(img, 140)
    for _ in range(2):
        img = saturate_contrastA(img, 50, 0.5)
    for _ in range(1): 
        img = saturate_contrastB(img, 140)
    for _ in range(2):
        img = saturate_contrastA(img, 50, 0.5)
    
    #img = output
#    cv2.imshow('resuilt', output)
#    cv2.waitKey()

    img = np.expand_dims(img, 0)
    
    img = img/255
    h = model.predict(img)
    # Dry && Normal && Oilly Intensity
    '''
    result = h.argmax()
    if result == 2: 
        return "Oilly"
    elif result == 1:
        return "Normal"
    elif result == 0:
        return "Dry"
    else:
        print("sum error occured on oil predictor")
        return None
    '''
    score = 32*h[0][0] + 64*h[0][1] + 96*h[0][2]
    score = (score - 34) / 61
    return 35 + int(score * 22)