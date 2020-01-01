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

    return glcm


def glcm_features(src, d_x=1, d_y=0, gray_level=16):
    h, w = src.shape
    glcm_0 = glcm(src, d_x, d_y, gray_level)
    # 对比度
    contrast = np.zeros((h, w), dtype=np.float64)
    # 能量
    asm = np.zeros((h, w), dtype=np.float64)
    # 熵值
    entropy = np.zeros((h, w), dtype=np.float64)
    # 均值
    mean = np.zeros((h, w), dtype=np.float64)

    for i in range(gray_level):
        for j in range(gray_level):
            asm += glcm_0[i][j] ** 2
            contrast += glcm_0[i][j] * (i - j) ** 2
            mean += glcm_0[i][j] * i
            if glcm_0[i][j] > 0.0:
                entropy -= glcm_0[i][j] * np.log(glcm_0[i][j])

    return contrast, asm, entropy, mean
  

if __name__ == '__main__':
    img = cv2.imread(r'../img/Lenna.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm_0 = glcm(img, -1, 1)
    cont, asm, ent, mean = glcm_features(img)

    plt.figure(figsize=(10, 4.5))
    fs = 15
    plt.subplot(2, 5, 1)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(img)
    plt.title('glcm', fontsize=fs)

    plt.subplot(2, 5, 2)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(mean)
    plt.title('mean', fontsize=fs)

    plt.subplot(2, 5, 3)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(cont)
    plt.title('contrast', fontsize=fs)

    plt.subplot(2, 5, 4)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(asm)
    plt.title('ASM', fontsize=fs)

    plt.subplot(2, 5, 5)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(ent)
    plt.title('entropy', fontsize=fs)

    plt.tight_layout(pad=0.5)
    # plt.savefig('img/output.jpg')
    plt.show()

