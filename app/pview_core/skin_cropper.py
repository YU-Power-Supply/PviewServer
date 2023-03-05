'''
    분류 : 전처리
    목적 : 이미지상의 얼굴영역 검출 및 추출 (서비스용)
'''

import cv2
import numpy as np
import mediapipe as mp

# Separate Face Part ####################################################################################################
rightCheek = [121, 47, 142, 203, 206, 216, 214, 192, 213, 147, 123, 117, 118, 119, 120]
leftCheek = [350, 277, 371, 423, 426, 436, 434, 416, 433, 376, 352, 346, 347, 348, 349]
rightEye = [226, 113, 124, 156, 143, 111, 117, 118, 119, 120, 232, 112, 26, 22, 23, 24, 110]
leftEye = [446, 342, 353, 383, 372, 340, 346, 347, 348, 349, 452, 341, 256, 252, 253, 254, 339]
forehead = [10, 338, 297, 332, 284, 334, 296, 336, 9, 107, 66, 105, 54, 103, 67, 109]
chin = [17, 406, 422, 430, 379, 400, 152, 176, 150, 210, 202, 182]
nose = [6, 122, 188, 217, 198, 209, 49, 48, 219, 218, 237, 44, 1, 274, 457, 438, 439, 278, 279, 429, 420, 437, 412, 351]

def create_landmark_img(img):
    height,width, _ = img.shape

    mp_holistic = mp.solutions.holistic # mediapipe solutions
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # Recolor Feed
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = cv2.flip(image, 1)
        image.flags.writeable = False
        
        # Make Detections
        # results = holistic.process(image)
        results = face_mesh.process(image)

        weight = 1.0
        if results.multi_face_landmarks:
            facial_landmarks = results.multi_face_landmarks[0]
            landmark_output = []
            # Total 468
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width * 1.0)
                y = int(pt1.y * height * weight)
                landmark_output.append([x,y])

            return landmark_output

        else:
            print('Error ... No result of facial landmark')

def cropImage(img, area, facial_landmark = []):
    height, width,  _ = img.shape
    mask = np.zeros((height, width, 3),dtype = np.uint8)
    if facial_landmark == []:
        print('Error ... no randmark data')
        return
    if area == 'rightCheek':
        rightCheek_poly = []
        for i in rightCheek:
            rightCheek_poly.append(facial_landmark[i])
        poly_area = np.array(rightCheek_poly, np.int32)
        
    elif area == 'leftCheek':
        leftCheek_poly = []
        for i in leftCheek:
            leftCheek_poly.append(facial_landmark[i]) 
        poly_area = np.array(leftCheek_poly, np.int32)

    elif area == 'rightEye':
        rightEye_poly = []
        for i in rightEye:
            rightEye_poly.append(facial_landmark[i]) 
        poly_area = np.array(rightEye_poly, np.int32)
        
    elif area == 'leftEye':
        leftEye_poly = []
        for i in leftEye:
            leftEye_poly.append(facial_landmark[i]) 
        poly_area = np.array(leftEye_poly, np.int32)
        
    elif area == 'forehead':
        forehead_poly = []
        for i in forehead:
            forehead_poly.append(facial_landmark[i]) 
        poly_area = np.array(forehead_poly, np.int32)
        
        
    elif area == 'chin':
        chin_poly = []
        for i in chin:
            chin_poly.append(facial_landmark[i]) 
        poly_area = np.array(chin_poly, np.int32)
        
    elif area == 'nose':
        nose_poly = []
        for i in nose:
            nose_poly.append(facial_landmark[i]) 
        poly_area = np.array(nose_poly, np.int32)
        
    ##### ##### ##### ##### ##### ##### ##### 마스크 생성부 ##### ##### ##### ##### ##### ##### ##### ##### 
    mask = cv2.fillPoly(mask, [poly_area], (255,255,255))
    if area == 'forehead':
        # 눈썹 제거
        b, g, r = cv2.split(img)
        b[b>75] = 255
        b[b<74] = 0
        g[g>110] = 255
        g[g<109] = 0

        mask_bin = np.bitwise_and(b, g)
        test_mask = cv2.merge((mask_bin, mask_bin, mask_bin))

        mask = np.bitwise_and(mask, test_mask)
        
    return mask


def skin_cropper(img, area='all'):
    # area = 'all'  #area type
    height , width, _ = img.shape

    # img = cv2.imread(os.path.join(data_dir, imgName))
    mask = np.zeros((height, width, 3),dtype = np.uint8)
    landmarkReturn = create_landmark_img(img)

    if area == 'rightCheek':
        mask = cropImage(img, area, facial_landmark = landmarkReturn)
    elif area == 'leftCheek':
        mask = cropImage(img, area, facial_landmark = landmarkReturn)
    elif area == 'rightEye':
        mask = cropImage(img, area, facial_landmark = landmarkReturn)
    elif area == 'leftEye':
        mask = cropImage(img, area, facial_landmark = landmarkReturn)
    elif area == 'forehead':
        mask = cropImage(img, area, facial_landmark = landmarkReturn)
    elif area == 'chin':
        mask = cropImage(img, area, facial_landmark = landmarkReturn)
    elif area == 'nose':
        mask = cropImage(img, area, facial_landmark = landmarkReturn)
    elif area == 'all':
        mask1 = cropImage(img, 'rightCheek', facial_landmark = landmarkReturn)
        mask2 = cropImage(img, 'leftCheek', facial_landmark = landmarkReturn)
        mask3 = cropImage(img, 'rightEye', facial_landmark = landmarkReturn)
        mask4 = cropImage(img, 'leftEye', facial_landmark = landmarkReturn)
        mask5 = cropImage(img, 'forehead', facial_landmark = landmarkReturn)
        mask6 = cropImage(img, 'nose', facial_landmark = landmarkReturn)
        mask7 = cropImage(img, 'chin', facial_landmark = landmarkReturn)
        mask_set = [mask1, mask2, mask3, mask4, mask5, mask6, mask7]

        for masking in mask_set:
            mask = cv2.bitwise_or(mask, masking)

    else : print("Some error Occured ... , Check area name")

    masked_img = cv2.bitwise_and(img, mask)

    mask = mask[landmarkReturn[10][1]:landmarkReturn[152][1], landmarkReturn[123][0]:landmarkReturn[352][0]]
    masked_img = masked_img[landmarkReturn[10][1]:landmarkReturn[152][1], landmarkReturn[123][0]:landmarkReturn[352][0]]

    return mask, masked_img

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

def line_detector(img):
    height, width, _ = img.shape
    alpha = 0.2
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