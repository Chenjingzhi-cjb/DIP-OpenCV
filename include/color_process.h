#ifndef DIP_OPENCV_COLOR_PROCESS_H
#define DIP_OPENCV_COLOR_PROCESS_H

#include "common.h"


using namespace std;
using namespace cv;


/**
 * @brief 彩色通道分离
 *
 * 调用 cv::split()
 *
 * @param src 输入图像
 * @return Color channel image group
 */
vector<Mat> colorChannelSpilt(Mat &src);

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
 * @brief 彩色分层
 *
 * @param src 输入图像；type: CV_8UC3
 * @param dst 输出图像
 * @param color_bgr 分层色彩值
 * @param range_r 分层色彩的范围
 * @return None
 */
void colorLayering(Mat &src, Mat &dst, const Vec3b& color_bgr, double range_r = 120);

// 彩色图像（RGB / HSI）的校正（对数变换 / 反对数变换 / 伽马变换）

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

// 彩色图像（RGB / HSI）的平滑和锐化


#endif //DIP_OPENCV_COLOR_PROCESS_H
