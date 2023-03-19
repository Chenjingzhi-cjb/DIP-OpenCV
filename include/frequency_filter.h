#ifndef DIP_OPENCV_FREQUENCY_FILTER_H
#define DIP_OPENCV_FREQUENCY_FILTER_H

#include <iostream>
#include <stdexcept>
#include <cmath>
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;


enum FreqKernel {
    idealLowPass,
    gaussLowPass,
    butterworthLowPass,
    idealHighPass,
    gaussHighPass,
    butterworthHighPass
};


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
 * @brief 理想低通频率滤波核函数，该核有振铃效应
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，平滑程度越大；D0 越大，平滑程度越小
 * @return Ideal low frequency kernel
 */
Mat idealLowPassFreqKernel(Size size, float sigma);

/**
 * @brief 高斯低通频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，平滑程度越大；D0 越大，平滑程度越小
 * @return Gaussian low frequency kernel
 */
Mat gaussLowPassFreqKernel(Size size, float sigma);

/**
 * @brief 巴特沃斯低通频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，平滑程度越大；D0 越大，平滑程度越小
 * @param order 即 n, 为阶数；n 越大，越接近理想滤波；n 越小，越接近高斯滤波
 * @return Butterworth low frequency kernel
 */
Mat bwLowPassFreqKernel(Size size, float sigma, int order);

/**
 * @brief 理想高通频率滤波核函数，该核有振铃效应
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，锐化程度越小；D0 越大，锐化程度越大
 * @return Ideal high pass frequency kernel
 */
Mat idealHighPassFreqKernel(Size size, float sigma);

/**
 * @brief 高斯高通频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，锐化程度越小；D0 越大，锐化程度越大
 * @return Gaussian high pass frequency kernel
 */
Mat gaussHighPassFreqKernel(Size size, float sigma);

/**
 * @brief 巴特沃斯高通频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，锐化程度越小；D0 越大，锐化程度越大
 * @param order 即 n, 为阶数；n 越大，越接近理想滤波；n 越小，越接近高斯滤波
 * @return Butterworth high pass frequency kernel
 */
Mat bwHighPassFreqKernel(Size size, float sigma, int order);

/**
 * @brief 频率域滤波
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param kernel 频率域滤波核；其尺寸应与处理图像的尺寸一致
 * @return None
 */
void frequencyFilter(Mat &src, Mat &dst, Mat &kernel);

/**
 * @brief 拉普拉斯频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @return Laplace frequency kernel
 */
Mat laplaceFreqKernel(Size size);

/**
 * @brief 拉普拉斯频率域图像增强
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @return Laplace frequency kernel
 */
void laplaceFreqImageEnhance(Mat &src, Mat &dst);


#endif //DIP_OPENCV_FREQUENCY_FILTER_H
