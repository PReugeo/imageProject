import  cv2
import numpy as np
import  sys
sys.path.append("image")
if __name__=="__main__":
    # 4 张不同的图片 sumer_0 和 winter_0 是同一个场景不同的季节
    # 实验结论 不同的图像（内容不一样 而不是仅仅是旋转和缩放的一样）
    # 不变矩的相差还是会很大
    #  数据来源 https://webdiis.unizar.es/~jmfacil/pr-nordland/
    live_summer = cv2.imread("summer_0.png",0)
    memory_winter = cv2.imread("winter_0.png",0)
    memory_winter_2 = cv2.imread("winter_85.png",0)
    m_live_summer = cv2.moments(live_summer)
    m_memory_winter = cv2.moments(memory_winter)
    m_memory_winter_2 = cv2.moments(memory_winter_2)
    hu_live_summer = cv2.HuMoments(m_live_summer)
    hu_memory_winter = cv2.HuMoments(m_memory_winter)
    hu_memory_winter_2 = cv2.HuMoments(m_memory_winter_2)
    # norml 归一化 先取绝对值（计算log）再取log)（去除高次方） 再取绝对值（去掉高次的负号）
    n_hu_live_summer = np.abs(np.log(np.abs(hu_live_summer)))
    n_hu_memory_winter = np.abs(np.log(np.abs(hu_memory_winter)))
    n_hu_memory_winter_2 = np.abs(np.log(np.abs(hu_memory_winter_2)))
    n_hu_live_summer_2 = np.abs(np.log(np.abs(cv2.HuMoments(cv2.moments(cv2.imread("summer_357.png", 0))))))
    print("summer ",n_hu_live_summer)
    print("winter ", n_hu_memory_winter)
    print("winter 2 ", n_hu_memory_winter_2)
    print("summer 2",n_hu_live_summer_2)
