import cv2
import numpy as np


def poreDetect(file_location):
    # 이미지 한 장 단위로 실행(테스트)
    img = cv2.imread(file_location)
    img = cv2.resize(img, (256, 256))

    img_b, img_g, img_r = cv2.split(img)  # B채널만 사용

    m_row = []
    m_column = []
    mrt = []
    for i in img_b:
        sortedList = np.sort(i)
        avr = np.mean(sortedList[int(255*0.1):int(255*0.9)])
        m_row.append(avr)
    rMax = max(m_row)
    m_row = m_row/rMax
    for i in m_row:
        mrt.append([i])

    for i in np.transpose(img_b):
        sortedList = np.sort(i)
        avr = np.mean(sortedList[int(255*0.1):int(255*0.9)])
        m_column.append(avr)
    cMax = max(m_column)
    m_column = [m_column/cMax]

    m_expact = np.dot(mrt, m_column)

    new_gray = img_b/m_expact
    for i in range(256):
        for j in range(256):
            if new_gray[i, j] > 255:
                new_gray[i, j] = 255

    new_gray = new_gray.astype(np.uint8)
    alpha = 0.2  # 얼마나 명암비를 올려줄 것인지에 대한 상수
    new_gray = np.clip((1+alpha)*new_gray - 128*alpha, 0, 255).astype(np.uint8)

    # hist = cv2.calcHist([new_gray], [0], None, [256], [0, 256])

    oneDimArr = sorted(np.ravel(new_gray, order='C'))
    referValue = oneDimArr[int(len(oneDimArr)*0.05)-1]
    # 120값 이미지마다 달라지니까 11.5퍼센트 정도로 찾을수 있는방법찾기
    ret, binaryImg = cv2.threshold(new_gray, referValue, 255, cv2.THRESH_BINARY_INV)

    zero_count = 0
    for i in range(256):
        for j in range(256):
            if binaryImg[i][j] == 0:
                zero_count += 1

    imgRatio = 1 - (zero_count/(256*256))
    print(f"모공 : {round(imgRatio*10,5)}")
    return round(imgRatio*10, 5)
