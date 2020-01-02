import  cv2
import numpy as np

if __name__=="__main__":
    live_summer = cv2.imread("imageProject/image/difrent_image_moments/summer_0.png",0)
    print(live_summer)
    # cv2.imshow("summer",live_summer)
    # memory_winter = cv2.imread("image/difrent_image_moments/winter_0.png",0)
    # memory_winter_2 = cv2.imread("image/difrent_image_moments/winter_85.png",0)
    # m_live_summer = cv2.moments(live_summer)
    # m_memory_winter = cv2.moments(memory_winter)
    # m_memory_winter_2 = cv2.moments(memory_winter_2)
    # hu_live_summer = cv2.HuMoments(m_live_summer)
    # hu_memory_winter = cv2.HuMoments(m_memory_winter)
    # hu_memory_winter_2 = cv2.HuMoments(m_memory_winter_2)
    # # norml 归一化 先取绝对值（计算log）再取log)（去除高次方） 再取绝对值（去掉高次的负号）
    # n_hu_live_summer = np.abs(np.log(np.abs(hu_live_summer)))
    # n_hu_memory_winter = np.abs(np.log(np.abs(hu_memory_winter)))
    # n_hu_memory_winter_2 = np.abs(np.log(np.abs(hu_memory_winter_2)))
    # n_hu_live_summer_2 = np.abs(np.log(np.abs(cv2.HuMoments(cv2.moments(cv2.imread("image/difrent_image_moments/summer_357.png", 0))))))
    # print("summer ",n_hu_live_summer)
    # print("winter ", n_hu_memory_winter)
    # print("winter 2 ", n_hu_memory_winter_2)
    # print("summer 2",n_hu_live_summer_2)
