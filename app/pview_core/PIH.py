'''
    분류 : 일반진단
    목적 : 피부 색소침착 검출 (서비스용)
'''

import cv2
import numpy as np
import tensorflow as tf
# import PIL
# from PIL import ImageOps # 사용되는 코드이니 절대 삭제하지 말것
from tensorflow import keras
import  copy

import numpy as np
import tensorflow as tf


def display_mask(val_preds, i):
  mask = np.argmax(val_preds[i], axis=-1)
  mask = np.expand_dims(mask, axis=-1)
  #img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask)) => 필요없는 코드
  #img = keras.preprocessing.image.array_to_img(mask) => 필요없는 코드
  return np.array(mask)

def test_model(img, model):#, img_size = (256, 256)):
  #img = cv2.resize(img, img_size)
  img = tf.expand_dims(img, axis= 0)
  # model = load_model(model_path)
  # model.summary()

  ## 추론 2.
  preds = model.predict(img)
  img2 = display_mask(preds, 0)
  del preds
  return 0

  return np.count_nonzero(img2)/len(img2)
  
def detect_pih(img, pihmodel):

  ### 1차 색소침착 : 멜라닌 지수를 이용한 색소침착 검출
  img_cie_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
  
  cie_l, _, cie_b = cv2.split(img_cie_lab)
  
  # individual typology angle
  ita = ((np.arctan((cie_l-50)/cie_b))*180/np.pi).astype(np.uint8)
  height, _ = ita.shape
  mean_ita = np.mean(ita)
  
  cut_st = 0.26
  cut_ed = 0.42
  
  # 강한색소침착
  ita_strong = ita.copy()
  ita_strong[0:int(height*cut_st)][:][ita_strong[0:int(height*cut_st)][:]<mean_ita*0.35] = 255
  ita_strong[int(height*cut_ed):][:][ita_strong[int(height*cut_ed):][:]<mean_ita*0.35] = 255
  
  # 중간색소침착
  ita_normal = ita.copy()
  ita_normal[0:int(height*cut_st)][:][ita_normal[0:int(height*cut_st)][:]<mean_ita*0.42] = 255
  ita_normal[int(height*cut_ed):][:][ita_normal[int(height*cut_ed):][:]<mean_ita*0.42] = 255

  # 약한 색소침착
  ita_weak = ita.copy()
  ita_weak[0:int(height*cut_st)][:][ita_weak[0:int(height*cut_st)][:]<mean_ita*0.60] = 255
  ita_weak[int(height*cut_ed):][:][ita_weak[int(height*cut_ed):][:]<mean_ita*0.60] = 255

  # 색소침착 합병
  img_pih = cv2.merge((ita_strong, ita_normal, ita_weak))

  # 눈가 그림자 제거
  img_pih[img_pih]

  # 1차 색소침착 : 영상처리 색소침착 
  STANDARD_VALUE = 200
  WLOHE_SCALE = 65536
  DIP_SCORE = 10*len(np.where(ita_strong>STANDARD_VALUE)[0])/WLOHE_SCALE + 5*len(np.where(ita_normal>STANDARD_VALUE)[0])/WLOHE_SCALE + len(np.where(ita_weak>STANDARD_VALUE)[0])/WLOHE_SCALE
  ### 2차 색소침착 : 시맨틱 세그맨테이션을 이용한 색소침착 검출
  S_SCORE = test_model(img, pihmodel)


  # 색소침착 점수 출력부
  P, S_SCORE = 0.9 , 0.1
  CONTRAST_BOUNDARY_VALUE = 0.3
  pih_score = DIP_SCORE * P + S_SCORE * (1-P)
  pih_score = 4*pih_score - 2*(pih_score - CONTRAST_BOUNDARY_VALUE)
  pih_score = np.clip((150 - np.clip(100*pih_score, 0, 100).astype(np.uint8)), 0, 100)
  
  return int(pih_score)+20

