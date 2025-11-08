#ifndef DIP_OPENCV_WAVELET_TRANSFORM_H
#define DIP_OPENCV_WAVELET_TRANSFORM_H

#include "common.h"


/**
 * @brief 离散小波变换
 *
 * @param src 输入图像；channels: 1
 * @param dst 输出图像
 * @param wname 小波基；["haar", "db1", "sym2"]
 * @param level 变换尺度
 * @return None
 */
void DWT(const cv::Mat &src, cv::Mat &dst, const std::string &wname, int level);

/**
 * @brief 离散小波逆变换
 *
 * @param src 输入图像；channels: 1
 * @param dst 输出图像
 * @param wname 小波基；["haar", "db1", "sym2"]
 * @param level 变换尺度
 * @return None
 */
void IDWT(const cv::Mat &src, cv::Mat &dst, const std::string &wname, int level);

/**
 * @brief 离散余弦变换
 *
 * 调用 cv::dct()
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @return None
 */
void DCT(const cv::Mat &src, cv::Mat &dst);

/**
 * @brief 离散余弦逆变换
 *
 * 调用 cv::idct()
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param original_size 原图像的尺寸，用于确保输出图像与原图像的尺寸一致
 * @return None
 */
void IDCT(const cv::Mat &src, cv::Mat &dst, const cv::Size &original_size);

/**
 * @brief 分块离散余弦变换
 *
 * 调用 cv::dct()
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param block_size 分块尺寸
 * @return None
 */
void blockDCT(const cv::Mat &src, cv::Mat &dst, int block_size = 8);

/**
 * @brief 分块离散余弦逆变换
 *
 * 调用 cv::idct()
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param original_size 原图像的尺寸，用于确保输出图像与原图像的尺寸一致
 * @param block_size 分块尺寸
 * @return None
 */
void blockIDCT(const cv::Mat &src, cv::Mat &dst, const cv::Size &original_size, int block_size = 8);


#endif //DIP_OPENCV_WAVELET_TRANSFORM_H
