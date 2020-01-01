import cv2
import numpy as np
# Hu Moments 胡不变矩 wiki： https://en.wikipedia.org/wiki/Image_moment
# 公式 https://docs.opencv.org/4.1.2/d3/dc0/group__imgproc__shape.html#gab001db45c1f1af6cbdbe64df04c4e944
# 代码 https://github.com/opencv/opencv/blob/b6a58818bb6b30a1f9d982b3f3f53228ea5a13c1/modules/imgproc/src/moments.cpp
# void cv::HuMoments( const Moments& m, double hu[7] )
# {
#     CV_INSTRUMENT_REGION();
#
#     double t0 = m.nu30 + m.nu12;
#     double t1 = m.nu21 + m.nu03;
#
#     double q0 = t0 * t0, q1 = t1 * t1;
#
#     double n4 = 4 * m.nu11;
#     double s = m.nu20 + m.nu02;
#     double d = m.nu20 - m.nu02;
#
#     hu[0] = s;
#     hu[1] = d * d + n4 * m.nu11;
#     hu[3] = q0 + q1;
#     hu[5] = d * (q0 - q1) + n4 * t0 * t1;
#
#     t0 *= q0 - 3 * q1;
#     t1 *= 3 * q0 - q1;
#
#     q0 = m.nu30 - 3 * m.nu12;
#     q1 = 3 * m.nu21 - m.nu03;
#
#     hu[2] = q0 * q0 + q1 * q1;
#     hu[4] = q0 * t0 + q1 * t1;
#     hu[6] = q1 * t0 - q0 * t1;
# }
if __name__ == "__main__":
    input_image = cv2.imread("image/lungs.jpg",0)
    # cut 1/4 image

    new_image = input_image[0:int(input_image.shape[0]/2),0:int(input_image.shape[1]/2)]
    print("new image size",new_image.shape)
    # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gafbbc470ce83812914a70abfb604f4326
    # rotate_45_image = cv2.rotate()
    #  旋转和平移图像的步骤都是一样的 先获取旋转矩阵 再将旋转矩阵和原图像输入到旋转函数warpAffine函数里面得到新的图像
    #  旋转和平移在旋转矩阵最大的不同就是图像的中心变不变 旋转不变（一般） 平移就变中心
    #  新图像和原图像相比 old（x,y）-> new（x,y）不一定是线性的 所以要用插值方法来计算这些不是线性的像素值映射
    # 常用的插值方法有 最近领插值 双线性插值、三次haermite插值哈哈哈哈
    rotate_45_matrix =cv2.getRotationMatrix2D((int(new_image.shape[1]/2),int(new_image.shape[0]/2)),-45,1)
    rotate_45_image = cv2.warpAffine(new_image,rotate_45_matrix,dsize=(new_image.shape[1],new_image.shape[0]))
    print("rotate_45_matrix",rotate_45_matrix)
    panning_15_pix_iamge_roata_matrax = cv2.getRotationMatrix2D((int(new_image.shape[1]/2)+15,int(new_image.shape[0]/2)+15),0,1)
    M = np.float32([[1, 0, 50], [0, 1, 50]])
    panning_15_pix_iamge = cv2.warpAffine(new_image,M,dsize=(new_image.shape[1],new_image.shape[0]))
    rotate_90_iamge = cv2.rotate(new_image, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    rotate_180_iamge = cv2.rotate(new_image,cv2.ROTATE_180)
    rotate_270_image = cv2.rotate(new_image, rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)
    resize_0_5_img = cv2.resize(new_image,(int(new_image.shape[0]/2),int(new_image.shape[1]/2)))
    # print("new_image shape",new_image.shape)
    # print("rotate 90 shape", rotate_90_iamge.shape)
    # print("rotate 180 shape", rotate_180_iamge.shape)
    # print("rotate 270 shape", rotate_270_image.shape)
    # calc the moments ofthe image
    # get 24 moments
    # 10 spatial moments 空间矩、原点矩
    # m00  m10  m01 m20 m11 m02 m30 m21 m12 m03
    #  7 central moments 七个中心矩
    # mu20 mu11 mu02 mu30 mu21 mu12 mu03
    #  7 central normalized moments个归一化中心矩
    # nu20 nu11 nu02 nu30 nu21 nu12 nu03
    #  本来应该都是10个的 but
    # mu00 = m00, nu00 = 1  nu10 = mu10 = mu01 = mu10 = 0, hence the values are not stored.
    #  原理 见论文（没看。。。）
    # 论文 https://pdfs.semanticscholar.org/afc2/e9d5dfbd666bf4dd34adeb78a17393c8ee64.pdf?_ga=2.259665167.462545856.1577780532-1866022657.1577780532
    # refrence https://docs.opencv.org/4.1.2/d8/d23/classcv_1_1Moments.html#a8b1b4917d1123abc3a3c16b007a7319b
    # https://github.com/opencv/opencv/blob/b6a58818bb6b30a1f9d982b3f3f53228ea5a13c1/modules/imgproc/src/moments.cpp
    m_ = cv2.moments(new_image)
    hu = cv2.HuMoments(m_)

    m_45 = cv2.moments(rotate_45_image)
    hu_45 = cv2.HuMoments(m_45)

    m_90 = cv2.moments(rotate_90_iamge)
    hu_90  = cv2.HuMoments(m_90)

    m_180 = cv2.moments(rotate_180_iamge)
    hu_180 = cv2.HuMoments(m_180)

    m_270 = cv2.moments(rotate_270_image)
    hu_270 = cv2.HuMoments(m_270)

    m_resize_0_5  =cv2.moments(resize_0_5_img)
    hu_resize_0_5 = cv2.HuMoments(m_resize_0_5)

    m_panning_15 = cv2.moments(panning_15_pix_iamge)
    hu_panning_15 = cv2.HuMoments(m_panning_15)
    print("new image hu ",hu)
    print("rotate 45 image hu ", hu_45)
    print("rotate 90 image hu ",hu_90)
    print("rotate 180 image hu ", hu_180)
    print("rotate 270 image hu ", hu_270)
    print("resize 1/2 image hu ", hu_resize_0_5)
    print("pamming y_15 x_50 image hu ", hu_panning_15)
    hu_list =np.array(hu)
    hu_list = np.append(hu_list, np.array(hu_45))
    hu_list = np.append(hu_list,np.array(hu_90))
    hu_list = np.append(hu_list, np.array(hu_180))
    hu_list = np.append(hu_list, np.array(hu_270))
    hu_list = np.append(hu_list, np.array(hu_resize_0_5))
    hu_list = np.append(hu_list, np.array(hu_panning_15))

    print(len(hu_list))
    # normal hu matrix
    hu_list = np.abs(hu_list)
    hu_list = np.log(hu_list)
    hu_list = np.abs(hu_list)

    print("hu_normal_matrix",hu_list)

    # cv2.imshow("input_image",input_image)
    cv2.imshow("cut_image", new_image)
    cv2.imshow("rotate 180", rotate_180_iamge)
    cv2.imshow("rotate 270", rotate_270_image)
    # cv2.imshow("rotate 90",rotate_90_iamge)
    # cv2.imshow("rotate 45 ", rotate_45_image)
    # cv2.imshow("panning 15 pix ", panning_15_pix_iamge)
    # cv2.imshow("resize 1/2 image",resize_0_5_img)


    cv2.waitKey(0000)