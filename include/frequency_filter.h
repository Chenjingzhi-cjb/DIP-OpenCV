#ifndef DIP_OPENCV_FREQUENCY_FILTER_H
#define DIP_OPENCV_FREQUENCY_FILTER_H

#include <iostream>
#include <stdexcept>
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;


/**
 * @brief 图像空间域转频率域
 *
 * 调用 cv::dft()
 *
 * @param src 输入图像；可以是实数或复数
 * @param dst 输出图像；其 size 和 type 由 flags 决定
 * @return None
 */
void spatialToFrequency(Mat &src, Mat &dst);

/**
 * @brief 图像频率域转空间域
 *
 * 调用 cv::idft()
 *
 * @param src 输入图像；可以是浮点实数或浮点复数
 * @param dst 输出图像；其 size 和 type 由 flags 决定
 * @param flags 转换标志；由 #DftFlags 组成
 * @param nonzeroRows 非零行参数；请参照 spatialToFrequency()
 * @return None
 */
void frequencyToSpatial(Mat &src, Mat &dst, int flags = 0, int nonzeroRows = 0);


#endif //DIP_OPENCV_FREQUENCY_FILTER_H
