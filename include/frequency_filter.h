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
 * @return None
 */
void spatialToFrequency(Mat &src, Mat &dst_complex);

/**
 * @brief 从频率域复数图像中分离出频率域实部幅值图像
 *
 * @param src_complex 输入图像，频率域复数图像
 * @param dst_magnitude 输出图像，频率域实数图像
 * @return None
 */
void splitFrequencyMagnitude(Mat &src_complex, Mat &dst_magnitude);

/**
 * @brief 图像频率域转空间域
 *
 * 调用 cv::idft()
 *
 * @param src_complex 输入图像，频率域复数图像
 * @param dst 输出图像，空间域实数图像；为确保输出图像与原图像的尺寸一致，应初始化并传入正确尺寸的 dst 对象，
 *        如：Mat dst = Mat::zeros(src.size(), src.depth());
 * @return None
 */
void frequencyToSpatial(Mat &src_complex, Mat &dst);

/**
 * @brief 空间域图像与频率域图像的转换演示
 *
 * @return None
 */
void domainTransformDemo();

/**
 * @brief 理想低通频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，模糊程度越大；D0 越大，模糊程度越小
 * @return Ideal low frequency kernel
 */
Mat idealLowFrequencyKernel(Size size, float sigma);

/**
 * @brief 高斯低通频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，模糊程度越大；D0 越大，模糊程度越小
 * @return Gauss low frequency kernel
 */
Mat gaussLowFrequencyKernel(Size size, float sigma);

/**
 * @brief 巴特沃斯低通频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，模糊程度越大；D0 越大，模糊程度越小
 * @param order 即 n, 为阶数；n 越大，越接近理想滤波；n 越小，越接近高斯滤波
 * @return Butterworth low frequency kernel
 */
Mat bwLowFrequencyKernel(Size size, float sigma, int order);

/**
 * @brief 平滑（低通）频率滤波
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param kernel 低通频率滤波核；其尺寸应与处理图像的尺寸一致
 * @return None
 */
void smoothFrequencyFilter(Mat &src, Mat &dst, Mat &kernel);


#endif //DIP_OPENCV_FREQUENCY_FILTER_H
