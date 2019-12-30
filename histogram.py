# 3.3.2 直方图正规化，P61
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('image/Lenna.png', 0)
cv2.imshow('original', img)
rows, cols = img.shape
print (rows, cols)

# img = np.array(img)
# 使用 np.amin   np.amax 获得图像(二维数组)中的最大、最小值
minOriginal = np.amin(img)
maxOriginal = np.amax(img)

ranges = maxOriginal - minOriginal

print ('原图中的最小亮度级: %d' %(minOriginal))
print ('原图中的最大亮度级: %d' %(maxOriginal))
new = np.zeros(img.shape, img.dtype)

# 对图像中的每一个像素(点)计算
'''
for x in range(0, cols):
    for y in range(0, rows):
        new[y, x] = ((img[y, x]-minOriginal)*255/ranges)
'''
# 根据P61页公式
for x in range(0, cols):
    for y in range(0, rows):
        new[y, x] =  255/ranges * (img[y, x]-minOriginal) + 0

print (new)
cv2.imshow('new', new)
print (img == new)

# 在判断条件中，不能使用 img==new
if (img.all() == new.all()):
    print ('这两幅图片相同')
elif (img.all() != new.all()):
    print ('这两幅图片不同')

# 绘制直方图，不使用opencv中的方法，直接使用matplotlib中的方法
plt.subplot(121), plt.hist(img.ravel(), 256, [0, 256]), plt.title('img')
plt.subplot(122), plt.hist(new.ravel(), 256, [0, 256]), plt.title('new')
plt.show()

cv2.waitKey(0)