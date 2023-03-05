'''
    분류 : 유틸리티
    목적 : 시각화 및 조도 제어
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

def pixelDetector(src):
    width, height = src.shape
    return len(src[src>0])/(width*height)

def plot2DImg(src):
    width, height = src.shape
    xx = np.arange(width)
    yy = np.arange(height)
    zz = []
    for x in range(width):
        for y in range(height):
            zz.append(src[x][y])

    X, Y = np.meshgrid(xx, yy)
    Z = np.asarray(zz, dtype=np.uint8).reshape((height, width))


    # Plotting 3D graph
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='magma', \
        edgecolor='none', antialiased=True,
        linewidth = 0)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_zlim(0, 255)


    # set view angles to get better plot
    ax.aziom = 70
    ax.elev = 50
    ax.dist = 10

    fig.colorbar(surf, shrink=0.5, aspect=15, pad=-0.05)
    plt.tight_layout()
    plt.show()

# 명도평준화(단일): ^avr 기준 ^alpha 기울기
def saturate_contrastA(Img, standard, alpha):
    func = (1+alpha) * Img - (alpha * standard) 
    dst = np.clip(func, 0, 255).astype(np.uint8)
    return dst

def display_image(image, name):
    window_name = name
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()