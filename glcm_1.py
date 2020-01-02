import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
d_x, d_y 决定多少度的共生矩阵
0度 d_x, d_y 取值为 1， 0
45度 1，1
90度 0，1
135度 -1，1

"""


def glcm(src, d_x, d_y, gray_level=16):
    src = src.copy()
    h, w = src.shape
    glcm = np.zeros([gray_level, gray_level])

    src = src.astype(np.float64)
    # 若灰度级大于 gray_level 则缩小灰度级
    max_level = src.max()
    if max_level > gray_level:
        src = src * (gray_level - 1) // max_level

    # 计算灰度共生矩阵
    for j in range(h - abs(d_y)):
        for i in range(w - abs(d_x)):
            rows = src[j][i].astype(int)
            cols = src[j + d_y][i + d_x].astype(int)
            glcm[rows][cols] += 1

    # 归一化灰度共生矩阵
    if d_x >= d_y:
        # 水平/垂直方向
        glcm = glcm / float(h * (w - 1))
    else:
        # 45/135
        glcm = glcm / float((h - 1) * (w - 1))

    return glcm


def glcm_features(src, d_x=1, d_y=0, gray_level=16):
    h, w = src.shape
    glcm_0 = glcm(src, d_x, d_y, gray_level)
    # 对比度
    contrast = 0.0
    # 能量
    asm = 0.0
    # 熵值
    entropy = 0.0
    # 均值
    mean = 0.0

    for i in range(gray_level):
        for j in range(gray_level):
            asm += glcm_0[i][j] ** 2
            contrast += glcm_0[i][j] * (i - j) ** 2
            mean += glcm_0[i][j] * i
            if glcm_0[i][j] > 0.0:
                entropy -= glcm_0[i][j] * np.log(glcm_0[i][j])

    return contrast, asm, entropy, mean
  

if __name__ == '__main__':
    img = cv2.imread(r'image/Fig0207(2Dsinewave).tif')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm_0 = glcm(img, 1, 0)
    cont, asm, ent, mean = glcm_features(img)

    print('对比度:', cont)
    print('角二阶矩:', asm)
    print('熵值:', ent)
    print('均值:', mean)

    plt.figure(figsize=(10, 4.5))
    fs = 15
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(glcm_0)
    plt.title('glcm', fontsize=fs)
    plt.show()

