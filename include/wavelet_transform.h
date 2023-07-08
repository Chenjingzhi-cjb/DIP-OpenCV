#ifndef DIP_OPENCV_WAVELET_TRANSFORM_H
#define DIP_OPENCV_WAVELET_TRANSFORM_H

#include "common.h"


using namespace std;
using namespace cv;


/**
 * @brief 离散小波变换
 *
 * @param src 输入图像；单通道
 * @param dst 输出图像
 * @param wname 小波基
 * @param level 变换层数
 * @return None
 */
void DWT(Mat &src, Mat &dst, const string &wname, int level);

/**
 * @brief 离散小波逆变换
 *
 * @param src 输入图像；单通道
 * @param dst 输出图像
 * @param wname 小波基
 * @param level 变换层数
 * @return None
 */
void IDWT(Mat &src, Mat &dst, const string &wname, int level);


#endif //DIP_OPENCV_WAVELET_TRANSFORM_H
