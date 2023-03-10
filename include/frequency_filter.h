#ifndef DIP_OPENCV_FREQUENCY_FILTER_H
#define DIP_OPENCV_FREQUENCY_FILTER_H

#include <iostream>
#include <stdexcept>
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;


/**
 * @brief 傅里叶图像象限变换
 *
 * @param image 处理图像
 * @return None
 */
void dftShift(Mat &image);

/**
 * @brief 图像空间域转频率域
 *
 * 调用 cv::dft()
 *
 * @param src 输入图像，空间域灰度图像
 * @param dst_complex 输出图像，频率域复数图像
 * @param dst_magnitude 输出图像，频率域实数图像
 * @return None
 */
void spatialToFrequency(Mat &src, Mat &dst_complex, Mat &dst_magnitude);

/**
 * @brief 图像频率域转空间域
 *
 * 调用 cv::idft()
 *
 * @param src 输入图像，频率域复数图像
 * @param dst 输出图像，空间域实数图像
 * @return None
 */
void frequencyToSpatial(Mat &src, Mat &dst);

/**
 * @brief 空间域图像与频率域图像的转换演示
 *
 * @return None
 */
void domainTransformDemo();


#endif //DIP_OPENCV_FREQUENCY_FILTER_H
