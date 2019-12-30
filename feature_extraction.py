import cv2
import numpy as np

img = cv2.imread('../img/Lenna_grey.png')

ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0] # 取第一条轮廓
M = cv2.moments(cnt)

# 画出轮廓
imgNew = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
print(M)

cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])

# 计算轮廓面积
area = cv2.contourArea(cnt)

# 计算周长
perimeter = cv2.arcLength(cnt, True)

# 轮廓的近似
epsilon = 0.02 * perimeter
approx = cv2.approxPolyDP(cnt, epsilon, True)
imgNew1 = cv2.drawContours(img, approx, -1, (0, 0, 255), 3)

cv2.imshow('lunkuo', imgNew)
cv2.imshow('approx_lunkuo', imgNew1)
cv2.waitKey(0)
