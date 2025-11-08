#ifndef DIP_OPENCV_GRAY_TRANSFORM_H
#define DIP_OPENCV_GRAY_TRANSFORM_H

#include "common.h"


/**
 * @brief BGR 转换为 GRAY
 *
 * 调用 cv::cvtColor()
 *
 * @param src 输入图像；type: CV_8UC3
 * @param dst 输出图像；type: CV_8UC1
 * @return None
 */
void bgrToGray(const cv::Mat &src, cv::Mat &dst);

/**
 * @brief 灰度线性缩放，缩放至 [0-255]
 *
 * 调用 cv:::normalize()
 *
 * @param src 输入图像；type: CV_8UC1
 * @param dst 输出图像
 * @return None
 */
void grayLinearScaleCV_8U(const cv::Mat &src, cv::Mat &dst);

/**
 * @brief 灰度反转（属于灰度线性变换）
 *
 * 调用 Mat::convertTo()
 *
 * @param src 输入图像；type: CV_8UC1
 * @param dst 输出图像
 * @return None
 */
void grayInvert(const cv::Mat &src, cv::Mat &dst);

/**
 * @brief 灰度对数变换，将输入图像中 范围较窄的低灰度值 映射为 范围较宽的灰度值级，
 *        同时将输入图像中 范围较宽的高灰度值 映射为 范围较窄的灰度值级，
 *        即拓展图像中的暗像素值、压缩亮像素值。
 *
 * @param src 输入图像；type: CV_8UC1
 * @param dst 输出图像
 * @return None
 */
void grayLog(const cv::Mat &src, cv::Mat &dst);

/**
 * @brief 灰度反对数变换，将输入图像中 范围较窄的高灰度值 映射为 范围较宽的灰度值级，
 *        同时将输入图像中 范围较宽的低灰度值 映射为 范围较窄的灰度值级，
 *        即拓展图像中的亮像素值、压缩暗像素值。
 *
 * @param src 输入图像；type: CV_8UC1
 * @param dst 输出图像
 * @return None
 */
void grayAntiLog(const cv::Mat &src, cv::Mat &dst);

/**
 * @brief 灰度伽马变换，也称幂律变换。当 0 < gamma < 1 时，
 *        将输入图像中 范围较窄的低灰度值 映射为 范围较宽的灰度值级，
 *        同时将输入图像中 范围较宽的高灰度值 映射为 范围较窄的灰度值级，
 *        即拓展图像中的暗像素值、压缩亮像素值；
 *        当 gamma > 1 时，效果相反。
 *
 * @param src 输入图像；type: CV_8UC1
 * @param dst 输出图像
 * @param gamma 指数；gamma > 0
 * @return None
 */
void grayGamma(const cv::Mat &src, cv::Mat &dst, float gamma);

/**
 * @brief 灰度对比度拉伸，根据 (输入灰度，输出灰度) -> (0, 0), (r1, s1), (r2, s2), (255, 255)
 *        来构建三段线性变换，使 (r1, r2) 的灰度值级拉伸映射到 (s1, s2) 的灰度值级。
 *
 * @param src 输入图像；type: CV_8UC1
 * @param dst 输出图像
 * @param r1 输入灰度值级 1
 * @param s1 输出灰度值级 1
 * @param r2 输入灰度值级 2
 * @param s2 输出灰度值级 2
 * @return None
 */
void grayContrastStretch(const cv::Mat &src, cv::Mat &dst, uint r1, uint s1, uint r2, uint s2);

/**
 * @brief 灰度值级分层，将 输入灰度值级区间 [r1, r2] 的值 置为 输出灰度值级 s，其他值置为 原值 / 零值。
 *
 * @param src 输入图像；type: CV_8UC1
 * @param dst 输出图像
 * @param r1 输入灰度值级 1
 * @param r2 输入灰度值级 2
 * @param s2 输出灰度值级
 * @param other_zero 设置其他值，true 为零值，false 为原值
 * @return None
 */
void grayLayering(const cv::Mat &src, cv::Mat &dst, uint r1, uint r2, uint s, bool other_zero);

/**
 * @brief 灰度比特平面分层，将输入图像根据灰度值的比特位进行分层。
 *
 * @param src 输入图像；type: CV_8UC1
 * @param dst 输出图像组
 * @return None
 */
void grayBitPlaneLayering(const cv::Mat &src, std::vector<cv::Mat> &dst);

/**
 * @brief 灰度直方图，即单通道直方图
 *
 * @param src 输入图像；type: CV_8UC1
 * @param mask 掩模
 * @param size 直方图的尺寸
 * @param color 直方图的颜色
 * @return gray histogram (single-channel histogram)
 */
cv::Mat grayHistogram(const cv::Mat &src, const cv::Mat &mask = cv::Mat(), cv::Size size = cv::Size(512, 400),
                      const cv::Scalar &color = cv::Scalar(255, 255, 255));

// 全局直方图均衡化
// OpenCV void cv::equalizeHist( cv::InputArray src, cv::OutputArray dst );

/**
 * @brief 局部直方图均衡化
 *
 * 调用 cv::createCLAHE()
 *
 * @param src 输入图像；type: CV_8UC1
 * @param dst 输出图像
 * @param clipLimit 对比度限制的阈值
 * @param tileGridSize 网格尺寸；输入图像将被分割成大小相等的矩形块
 * @return None
 */
void localEqualizeHist(const cv::Mat &src, cv::Mat &dst, double clipLimit = 40.0,
                       cv::Size tileGridSize = cv::Size(8, 8));

/**
 * @brief 直方图规定化
 *
 * @param src 输入原始图像；type: CV_8UC1
 * @param dst 输出图像
 * @param refer 输入参考图像；type: CV_8UC1
 * @return None
 */
void matchHist(const cv::Mat &src, cv::Mat &dst, const cv::Mat &refer);

/**
 * @brief 阴影校正
 *
 * @param src 输入图像；type: CV_8UC1
 * @param dst 输出图像
 * @param k1 卷积核尺寸 与 图像尺寸 的比例系数；取值范围为 (0, 0.5]，系数越大，模糊程度越高
 * @param k2 卷积核尺寸 与 sigma 的比例系数；取值范围为 (0, 6]
 * @return None
 */
void shadingCorrection(const cv::Mat &src, cv::Mat &dst, float k1 = 0.25, float k2 = 6);


#endif //DIP_OPENCV_GRAY_TRANSFORM_H
