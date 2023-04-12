#ifndef DIP_OPENCV_IMAGE_NOISE_H
#define DIP_OPENCV_IMAGE_NOISE_H

#include <iostream>
#include <stdexcept>
#include <cmath>
#include "opencv2/opencv.hpp"
#include "gray_transform.h"


using namespace std;
using namespace cv;


/**
 * @brief 添加高斯噪声
 *
 * 调用 rng.fill(RNG::NORMAL)
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param mean 高斯分布的均值
 * @param sigma 高斯分布的标准差
 * @return None
 */
void addNoiseGauss(Mat &src, Mat &dst, int mean, int sigma);

/**
 * @brief 添加平均噪声
 *
 * 调用 rng.fill(RNG::UNIFORM)
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param lower 平均分布的下限（闭区间）
 * @param upper 平均分布的上限（开区间）
 * @return None
 */
void addNoiseMean(Mat &src, Mat &dst, int lower, int upper);

/**
 * @brief 添加瑞利噪声
 *
 * @param src 输入图像；type: CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3
 * @param dst 输出图像
 * @param sigma 瑞利分布的标准差
 * @return None
 */
void addNoiseRayleigh(Mat &src, Mat &dst, double sigma);

/**
 * @brief 添加伽马(爱尔兰）噪声
 *
 * @param src 输入图像；type: CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3
 * @param dst 输出图像
 * @param sigma 伽马分布的尺度参数
 * @param alpha 伽马分布的形状参数
 * @param beta 伽马分布的形状参数
 * @return None
 */
void addNoiseGamma(Mat &src, Mat &dst, double sigma, double alpha, double beta);

/**
 * @brief 添加指数噪声
 *
 * @param src 输入图像；type: CV_8UC1, CV_8UC3, CV_32FC1, CV_32FC3
 * @param dst 输出图像
 * @param lambda 指数分布的速率参数
 * @return None
 */
void addNoiseExp(Mat &src, Mat &dst, double lambda);

/**
 * @brief 添加椒盐噪声
 *
 * @param src 输入图像；channels: 1, 3
 * @param dst 输出图像
 * @param noise_level 噪声分层值；取值范围为 (0, 1)；
 *        含义为 0 <= random < noise_level/2 => salt, noise_level/2 <= random < noise_level => pepper
 * @param salt_value 盐噪声值（即白色噪声）；默认 depth() 为 CV_8U，则 salt_value 为 255
 * @param pepper_value 椒噪声值（即黑色噪声）；默认 depth() 为 CV_8U，则 pepper_value 为 0
 * @return None
 */
void addNoiseSaltPepper(Mat &src, Mat &dst, double noise_level, double salt_value = 255, double pepper_value = 0);


#endif //DIP_OPENCV_IMAGE_NOISE_H
