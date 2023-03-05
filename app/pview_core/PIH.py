import cv2
import numpy as np
import tensorflow as tf
import PIL
from tensorflow import keras
import  copy

from tensorflow.keras.models import load_model



def detect_pih(img, model_path):

  # 1차 색소침착 검출
  img_cie_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
  
  cie_l, _, cie_b = cv2.split(img_cie_lab)
  
  # individual typology angle
  ita = ((np.arctan((cie_l-50)/cie_b))*180/np.pi).astype(np.uint8)
  height, _ = ita.shape
  
  cut_st = 0.26
  cut_ed = 0.42
  
  # 강한색소침착
  ita_strong = copy.deepcopy(ita)
  ita_strong[0:int(height*cut_st)][:][ita_strong[0:int(height*cut_st)][:]<np.mean(ita)*0.35] = 255
  ita_strong[int(height*cut_ed):][:][ita_strong[int(height*cut_ed):][:]<np.mean(ita)*0.35] = 255
  
  # 중간색소침착
  ita_normal = copy.deepcopy(ita)
  ita_normal[0:int(height*cut_st)][:][ita_normal[0:int(height*cut_st)][:]<np.mean(ita)*0.42] = 255
  ita_normal[int(height*cut_ed):][:][ita_normal[int(height*cut_ed):][:]<np.mean(ita)*0.42] = 255

  # 약한 색소침착
  ita_weak = copy.deepcopy(ita)
  ita_weak[0:int(height*cut_st)][:][ita_weak[0:int(height*cut_st)][:]<np.mean(ita)*0.60] = 255
  ita_weak[int(height*cut_ed):][:][ita_weak[int(height*cut_ed):][:]<np.mean(ita)*0.60] = 255

  # 색소침착 합병
  img_pih = cv2.merge((ita_strong, ita_normal, ita_weak))

  # # 눈가 그림자 제거
  img_pih[img_pih]

  # 2차 색소침착 검출 
  # test_model(img_origin, model_path)

  # 색소침착 점수 출력부
  P, S_SCORE = 1 , 0
  pih_score = cv2.mean(3*cv2.mean(ita_strong) + 2*cv2.mean(ita_normal) + cv2.mean(ita_weak))[0] * P + S_SCORE * (1-P)
  
  pih_score = (100 - np.clip(pih_score, 0, 100).astype(np.uint8))
  
  return pih_score