#ifndef DIP_OPENCV_COLOR_PROCESS_H
#define DIP_OPENCV_COLOR_PROCESS_H

#include "common.h"


/**
 * @brief 彩色通道分离
 *
 * 调用 cv::split()
 *
 * @param src 输入图像
 * @return Color Image Channels
 */
vector<Mat> colorChannelSpilt(Mat &src);

/**
 * @brief 彩色通道合并
 *
 * 调用 cv::merge()
 *
 * @param channels 输入图像通道
 * @return Color Image
 */
Mat colorChannelMerge(vector<Mat> &channels);

/**
 * @brief BGR 转换为 HSI
 *
 * @param src 输入图像；type: CV_8UC3
 * @param dst 输出图像
 * @return None
 */
void bgrToHsi(Mat &src, Mat &dst);

/**
 * @brief HSI 转换为 BGR
 *
 * @param src 输入图像；type: CV_8UC3
 * @param dst 输出图像
 * @return None
 */
void hsiToBgr(Mat &src, Mat &dst);

/**
 * @brief 伪彩色处理
 *
 * 调用 cv::applyColorMap()
 *
 * @param src 输入图像；type: CV_8UC1(grayscale) or CV_8UC3(colored)
 * @param dst 输出图像
 * @param color 色彩；请参阅 #ColormapTypes
 * @return None
 */
void pseudoColor(Mat &src, Mat &dst, ColormapTypes color = COLORMAP_JET);

/**
 * @brief 补色处理，即彩色反转
 *
 * 调用 Mat::convertTo()
 *
 * @param src 输入图像；type: CV_8UC3
 * @param dst 输出图像
 * @return None
 */
void complementaryColor(Mat &src, Mat &dst);

/**
 * @brief 彩色分层（球形)
 *
 * @param src 输入图像；type: CV_8UC3
 * @param dst 输出图像
 * @param color_center 分层色彩中心值
 * @param range_radius 分层范围半径值
 * @return None
 */
void colorLayering(Mat &src, Mat &dst, const Vec3b& color_center, double range_radius = 120);

// 彩色图像（RGB / HSI）的校正（对数变换 / 反对数变换 / 伽马变换）：参考灰度

/**
 * @brief 彩色全局直方图均衡化（不建议使用）
 *
 * 调用 cv::equalizeHist()
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @return None
 */
void colorEqualizeHist(Mat &src, Mat &dst);


#endif //DIP_OPENCV_COLOR_PROCESS_H
