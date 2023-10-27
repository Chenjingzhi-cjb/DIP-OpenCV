#ifndef DIP_OPENCV_FREQUENCY_FILTER_H
#define DIP_OPENCV_FREQUENCY_FILTER_H

#include "common.h"


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
 * @param dst_complex 输出图像，即 dft 处理后的频率域复数图像
 * @return None
 */
void spatialToFrequency(Mat &src, Mat &dst_complex);

/**
 * @brief 从频率域复数图像中分离出频率域实部幅值图像
 *
 * @param src_complex 输入图像，频率域复数图像
 * @param dst_magnitude 输出图像，频率域实数图像；缩放为 CV_32F [0, 1]
 * @return None
 */
void splitFrequencyMagnitude(Mat &src_complex, Mat &dst_magnitude);

/**
 * @brief 图像频率域转空间域
 *
 * 调用 cv::idft()
 *
 * @param src_complex 输入图像，频率域复数图像
 * @param dst 输出图像，即 idft 处理后的空间域实数图像；
 *        为确保输出图像与原图像的尺寸一致，应先初始化并传入正确尺寸的 dst 对象，
 *        如：Mat dst = Mat::zeros(src.size(), src.depth());
 * @return None
 */
void frequencyToSpatial(Mat &src_complex, Mat &dst);

/**
 * @brief 理想低通频率滤波核函数，该核有振铃效应
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，平滑程度越大；D0 越大，平滑程度越小
 * @return Ideal low frequency kernel
 */
Mat idealLowPassFreqKernel(Size size, int sigma);

/**
 * @brief 高斯低通频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，平滑程度越大；D0 越大，平滑程度越小
 * @return Gaussian low frequency kernel
 */
Mat gaussLowPassFreqKernel(Size size, int sigma);

/**
 * @brief 巴特沃斯低通频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，平滑程度越大；D0 越大，平滑程度越小
 * @param order 即 n，为阶数；n 越大，越接近理想滤波；n 越小，越接近高斯滤波
 * @return Butterworth low frequency kernel
 */
Mat bwLowPassFreqKernel(Size size, int sigma, int order);

/**
 * @brief 理想高通频率滤波核函数，该核有振铃效应
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，锐化程度越小；D0 越大，锐化程度越大
 * @return Ideal high pass frequency kernel
 */
Mat idealHighPassFreqKernel(Size size, int sigma);

/**
 * @brief 高斯高通频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，锐化程度越小；D0 越大，锐化程度越大
 * @return Gaussian high pass frequency kernel
 */
Mat gaussHighPassFreqKernel(Size size, int sigma);

/**
 * @brief 巴特沃斯高通频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为截止频率；D0 越小，锐化程度越小；D0 越大，锐化程度越大
 * @param order 即 n，为阶数；n 越大，越接近理想滤波；n 越小，越接近高斯滤波
 * @return Butterworth high pass frequency kernel
 */
Mat bwHighPassFreqKernel(Size size, int sigma, int order);

/**
 * @brief 高频增强滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为高通滤波截止频率
 * @param k1 直流项
 * @param k2 高频贡献率
 * @return High frequency emphasis kernel
 */
Mat highFreqEmphasisKernel(Size size, int sigma, float k1 = 1, float k2 = 1);

/**
 * @brief 同态增强滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param sigma 即 D0，为高通滤波截止频率
 * @param gamma_h 高频值；gamma_h >= 1
 * @param gamma_l 低频值；0 < gamma_l < 1
 * @param c 传递函数偏斜度常数；类似于巴特沃斯的阶数
 * @return Homomorphic emphasis kernel
 */
Mat homomorphicEmphasisKernel(Size size, int sigma, float gamma_h, float gamma_l, int c);

/**
 * @brief 理想带阻频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param c0 频带中心
 * @param width 频带宽度
 * @return Ideal band reject frequency kernel
 */
Mat idealBandRejectFreqKernel(Size size, int C0, int width);

/**
 * @brief 高斯带阻频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param c0 频带中心
 * @param width 频带宽度
 * @return Gauss band reject frequency kernel
 */
Mat gaussBandRejectFreqKernel(Size size, int C0, int width);

/**
 * @brief 巴特沃斯带阻频率滤波核函数
 *
 * @param size 滤波核尺寸；应与处理图像的尺寸一致，src.size()
 * @param c0 频带中心
 * @param width 频带宽度
 * @param order 即 n，为阶数；n 越大，越接近理想滤波；n 越小，越接近高斯滤波
 * @return Butterworth band reject frequency kernel
 */
Mat bwBandRejectFreqKernel(Size size, int C0, int width, int order);

// Mat notchBandRejectFreqKernel() 陷波带阻滤波核（定制化，主要用于处理周期噪声）

/**
 * @brief 频率域滤波
 *
 * @param src 输入图像；注意 type 应为 CV_8UC1
 * @param dst 输出图像；缩放为 CV_8U [0, 255]
 * @param kernel 频率域滤波核；其尺寸应与处理图像的尺寸一致
 * @param rm_negative 负值移除标志位；true - 移除负值，false - 保留负值
 * @return None
 */
void frequencyFilter(Mat &src, Mat &dst, Mat &kernel, bool rm_negative = false);

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
 * @param src 输入图像；注意 type 应为 CV_8UC1
 * @param dst 输出图像
 * @return None
 */
void freqSharpenLaplace(Mat &src, Mat &dst);

/**
 * @brief 频率域滤波（复数乘法版）
 *
 * 调用 cv::mulSpectrums()
 *
 * @param src 输入图像；注意 type 应为 CV_8UC1
 * @param dst 输出图像；缩放为 CV_8U [0, 255]
 * @param kernel 频率域滤波核；其尺寸应与处理图像的尺寸一致；注意 type 应为 CV_32FC2
 * @param rm_negative 负值移除标志位；true - 移除负值，false - 保留负值
 * @return None
 */
void frequencyFilterPlMul(Mat &src, Mat &dst, Mat &kernel, bool rm_negative = false);

/**
 * @brief 最优陷波滤波
 *
 * @param src 输入图像；注意 type 应为 CV_8UC1
 * @param dst 输出图像
 * @param nbp_kernel 陷波带通滤波核；用于获取噪声图像，其尺寸应与处理图像的尺寸一致
 * @param opt_ksize 优化滤波核尺寸；用于优化滤波图像，必须为正奇数
 * @return None
 */
void bestNotchFilter(Mat &src, Mat &dst, Mat &nbp_kernel, Size opt_ksize);


#endif //DIP_OPENCV_FREQUENCY_FILTER_H
