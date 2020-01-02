import cv2
import numpy as np
import random
import math
img = np.zeros((500, 500), np.uint8)
img.fill(125)
for i in range(50, 100):
    for j in range(50, 200):
        img[i][j] = random.randint(150,255)

for i in range(0,250):
    for j in range(250,500):
        if (i-125)*(i-125)+(j-375)*(j-375) <= 10000:
            img[i][j] = 255

cv2.ellipse(img, (125, 375), (120, 60), 0, 0, 360, 255, 1)    # 画椭圆
cv2.imshow('1', img)
cv2.imwrite('src.png',img)
# cv2.waitKey()


def getDis(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
    a = lineY2-lineY1
    b = lineX1-lineX2
    c = lineX2*lineY1-lineX1*lineY2
    dis = (math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b, 0.5))
    return dis

def pointDis(x1,y1,x2,y2):
    return ((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1))**0.5

def areas(x1,y1,x2,y2,x3,y3):
    return(pointDis(x1,y1,x2,y2)*getDis(x3,y3,x1,y1,x2,y2)/2)

ret, thresh = cv2.threshold(img, 125, 255, 0)
cv2.imwrite('thresh.png', thresh)
cv2.imshow('2', thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for i in range(1, len(contours)):
    print(i)
    cnt = contours[i]
    rect = cv2.minAreaRect(cnt)
    M = cv2.moments(cnt)    # 计算矩
    # print(M)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print('重心：', cx, cy)
    min = math.inf
    min_i = 0
    min_j = 0
    for i in range(500):
        for j in range(500):
            sum = 0
            if (i - cx)*(i - cx) + (j - cy)*(j - cy) == 10000:
                for c0 in cnt[:][:]:
                    sum += getDis(c0[0][0], c0[0][1], cx, cy, i, j)
                if sum < min:
                    min = sum
                    min_i = i
                    min_j = j

    cv2.line(img, (cx, cy), (min_i, min_j), (0, 0, 0), 3, 8)
    area = cv2.contourArea(cnt)    # 计算轮廓面积
    print(area)
    lens = 0.0
    area=0.0
    for i in range(len(cnt)-1):
        lens += pointDis(cnt[i][0][0], cnt[i][0][1],cnt[i+1][0][0], cnt[i+1][0][1])
        area+=areas(cnt[i][0][0], cnt[i][0][1],cnt[i+1][0][0], cnt[i+1][0][1],cx,cy)
    circle = 4.0*np.pi*area/(lens*lens)  # 计算圆形度
    rectange = area/(rect[1][0]*rect[1][1])     # 计算矩形度
    print('圆形度：', circle, ' 矩形度', rectange)
    print('面积：', area, '周长：', lens)

cv2.imshow('final', img)
cv2.imwrite('final.png', img)
cv2.waitKey(0)
