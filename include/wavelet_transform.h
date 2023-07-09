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
 * @param level 变换尺度
 * @return None
 */
void DWT(Mat &src, Mat &dst, const string &wname, int level);

/**
 * @brief 离散小波逆变换
 *
 * @param src 输入图像；单通道
 * @param dst 输出图像
 * @param wname 小波基
 * @param level 变换尺度
 * @return None
 */
void IDWT(Mat &src, Mat &dst, const string &wname, int level);

/**
 * @brief 离散余弦变换
 *
 * 调用 cv::dct()
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @return None
 */
void DCT(Mat &src, Mat &dst);

/**
 * @brief 离散余弦逆变换
 *
 * 调用 cv::idct()
 *
 * @param src 输入图像
 * @param dst 输出图像
 *        为确保输出图像与原图像的尺寸一致，应先初始化并传入正确尺寸的 dst 对象，
 *        如：Mat dst = Mat::zeros(src.size(), src.type());
 * @return None
 */
void IDCT(Mat &src, Mat &dst);

/**
 * @brief 离散余弦变换及逆变换演示
 *
 * @return None
 */
void DCTDemo();

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
void blockDCT(Mat &src, Mat &dst, int block_size = 8);

/**
* @brief 分块离散余弦逆变换
*
* 调用 cv::idct()
*
* @param src 输入图像
* @param dst 输出图像
*        为确保输出图像与原图像的尺寸一致，应先初始化并传入正确尺寸的 dst 对象，
*        如：Mat dst = Mat::zeros(src.size(), src.type());
* @param block_size 分块尺寸
* @return None
*/
void blockIDCT(Mat &src, Mat &dst, int block_size = 8);

/**
 * @brief 分块离散余弦变换及逆变换演示
 *
 * @return None
 */
void blockDCTDemo();


#endif //DIP_OPENCV_WAVELET_TRANSFORM_H
