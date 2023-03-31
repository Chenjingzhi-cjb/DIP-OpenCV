#ifndef DIP_OPENCV_GRAY_TRANSFORM_H
#define DIP_OPENCV_GRAY_TRANSFORM_H

#include <iostream>
#include <stdexcept>
#include "opencv2/opencv.hpp"
#include <cmath>
#include <vector>


using namespace std;
using namespace cv;


/**
 * @brief BGR 转换为 GRAY
 *
 * 调用 cv::cvtColor()
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @return None
 */
void bgrToGray(Mat &src, Mat &dst);

/**
 * @brief 灰度线性缩放，缩放至 [0-255]
 *
 * 调用 Mat::normalize()
 *
 * @param src 输入图像；注意 depth 应为 CV_8U
 * @param dst 输出图像
 * @return None
 */
void grayLinearScaleCV_8U(Mat &src, Mat &dst);

/**
 * @brief 灰度反转（属于灰度线性变换）
 *
 * 调用 Mat::convertTo()
 *
 * @param src 输入图像；注意 depth 应为 CV_8U
 * @param dst 输出图像
 * @return None
 */
void grayInvert(Mat &src, Mat &dst);

/**
 * @brief 灰度对数变换，将输入图像中 范围较窄的低灰度值 映射为 范围较宽的灰度值级，
 *        同时将输入图像中 范围较宽的高灰度值 映射为 范围较窄的灰度值级，
 *        即拓展图像中的暗像素值、压缩亮像素值。
 *
 * @param src 输入图像；注意 depth 应为 CV_8U
 * @param dst 输出图像
 * @param k 系数
 * @return None
 */
void grayLog(Mat &src, Mat &dst, float k);

/**
 * @brief 灰度反对数变换，将输入图像中 范围较窄的高灰度值 映射为 范围较宽的灰度值级，
 *        同时将输入图像中 范围较宽的低灰度值 映射为 范围较窄的灰度值级，
 *        即拓展图像中的亮像素值、压缩暗像素值。
 *
 * @param src 输入图像；注意 depth 应为 CV_8U
 * @param dst 输出图像
 * @param k 系数
 * @return None
 */
void grayAntiLog(Mat &src, Mat &dst, float k);

/**
 * @brief 灰度伽马变换，也称幂律变换。当 0 < gamma < 1 时，
 *        将输入图像中 范围较窄的低灰度值 映射为 范围较宽的灰度值级，
 *        同时将输入图像中 范围较宽的高灰度值 映射为 范围较窄的灰度值级，
 *        即拓展图像中的暗像素值、压缩亮像素值；
 *        当 gamma > 1 时，效果相反。
 *
 * @param src 输入图像；注意 depth 应为 CV_8U
 * @param dst 输出图像
 * @param k 系数
 * @param gamma 指数；gamma > 0
 * @return None
 */
void grayGamma(Mat &src, Mat &dst, float k, float gamma);

/**
 * @brief 灰度对比度拉伸，根据 (输入灰度，输出灰度) -> (0, 0), (r1, s1), (r2, s2), (255, 255)
 *        来构建三段线性变换，使 (r1, r2) 的灰度值级拉伸映射到 (s1, s2) 的灰度值级。
 *
 * @param src 输入图像；注意 depth 应为 CV_8U
 * @param dst 输出图像
 * @param r1 输入灰度值级 1
 * @param s1 输出灰度值级 1
 * @param r2 输入灰度值级 2
 * @param s2 输出灰度值级 2
 * @return None
 */
void grayContrastStretch(Mat &src, Mat &dst, uint r1, uint s1, uint r2, uint s2);

/**
 * @brief 灰度值级分层，将 输入灰度值级区间 [r1, r2] 的值 置为 输出灰度值级 s，其他值置为 原值 / 零值。
 *
 * @param src 输入图像；注意 depth 应为 CV_8U
 * @param dst 输出图像
 * @param r1 输入灰度值级 1
 * @param r2 输入灰度值级 2
 * @param s2 输出灰度值级
 * @param other_zero 设置其他值，true 为零值，false 为原值
 * @return None
 */
void grayLayering(Mat &src, Mat &dst, uint r1, uint r2, uint s, bool other_zero);

/**
 * @brief 灰度比特平面分层，将输入图像根据灰度值的比特位进行分层。
 *
 * @param src 输入图像；注意 depth 应为 CV_8U
 * @param dst 输出图像组
 * @return None
 */
void grayBitPlaneLayering(Mat &src, vector<Mat> &dst);

/**
 * @brief 灰度直方图，即单通道直方图
 *
 * @param src 输入图像；注意 type 应为 CV_8UC1
 * @param dst 输出图像
 * @param size 输出直方图的尺寸
 * @param color 输出直方图的颜色
 * @return None
 */
void grayHistogram(Mat &src, Mat &dst, Size size = Size(512, 400), const Scalar &color = Scalar(255, 255, 255));

// 全局直方图均衡化
// OpenCV void equalizeHist( InputArray src, OutputArray dst );

/**
 * @brief 局部直方图均衡化
 *
 * 调用 cv::createCLAHE()
 *
 * @param src 输入图像；注意 type 应为 CV_8UC1
 * @param dst 输出图像
 * @param clipLimit 对比度限制的阈值
 * @param tileGridSize 网格尺寸；输入图像将被分割成大小相等的矩形块
 * @return None
 */
void localEqualizeHist(Mat &src, Mat &dst, double clipLimit = 40.0, Size tileGridSize = Size(8, 8));

/**
 * @brief 直方图规定化
 *
 * @param src 输入原始图像；注意 type 应为 CV_8UC1
 * @param dst 输出图像
 * @param refer 输入参考图像；注意 type 应为 CV_8UC1
 * @return None
 */
void matchHist(Mat &src, Mat &dst, Mat &refer);


#endif //DIP_OPENCV_GRAY_TRANSFORM_H
