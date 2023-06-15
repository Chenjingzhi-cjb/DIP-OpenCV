#ifndef DIP_OPENCV_MORPHOLOGICAL_H
#define DIP_OPENCV_MORPHOLOGICAL_H

#include "common.h"


using namespace std;
using namespace cv;


// 构建（形态学）结构元
// OpenCV Mat getStructuringElement(int shape, Size ksize, Point anchor = Point(-1,-1));

/**
 * @brief 形态学腐蚀
 *
 * 调用 cv::erode() / cv::morphologyEx(cv::MORPH_ERODE)
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param kernel 卷积核；尺寸必须为正奇数，像素值通常为 0 / 1
 * @return None
 */
void morphologyErode(Mat &src, Mat &dst, const Mat &kernel);

/**
 * @brief 形态学膨胀
 *
 * 调用 cv::dilate() / cv::morphologyEx(cv::MORPH_DILATE)
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param kernel 卷积核；尺寸必须为正奇数，像素值通常为 0 / 1
 * @return None
 */
void morphologyDilate(Mat &src, Mat &dst, const Mat &kernel);

/**
 * @brief 形态学开运算
 *
 * 调用 cv::morphologyEx(cv::MORPH_OPEN)
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param kernel 卷积核；尺寸必须为正奇数，像素值通常为 0 / 1
 * @return None
 */
void morphologyOpen(Mat &src, Mat &dst, const Mat &kernel);

/**
 * @brief 形态学闭运算
 *
 * 调用 cv::morphologyEx(cv::MORPH_CLOSE)
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param kernel 卷积核；尺寸必须为正奇数，像素值通常为 0 / 1
 * @return None
 */
void morphologyClose(Mat &src, Mat &dst, const Mat &kernel);

/**
 * @brief 形态学击中击不中变换
 *
 * 调用 cv::morphologyEx(cv::MORPH_HITMISS)
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param fore_kernel 前景卷积核（击中）；尺寸必须为正奇数，像素值通常为 0 / 1
 * @param back_kernel 背景卷积核（击不中）；尺寸必须为正奇数，像素值通常为 0 / 1
 * @return None
 */
void morphologyHMT(Mat &src, Mat &dst, const Mat &fore_kernel, const Mat &back_kernel);

/**
 * @brief 边界提取
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @param size 卷积核尺寸；通常取决于所需的边界尺寸
 * @return None
 */
void boundaryExtract(Mat &src, Mat &dst, int size);

/**
 * @brief 孔洞填充
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @param start 孔洞起始点图像；type: CV_8UC1，二值图
 * @return None
 */
void holeFill(Mat &src, Mat &dst, Mat &start);

/**
 * @brief 提取连通分量
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @return None
 */
void extractConnected(Mat &src, Mat &dst);

// 凸壳、细化、粗化、骨架、裁剪


#endif //DIP_OPENCV_MORPHOLOGICAL_H
