#ifndef DIP_OPENCV_IMAGE_SEGMENTATION_H
#define DIP_OPENCV_IMAGE_SEGMENTATION_H

#include "common.h"
#include "gray_transform.h"


/**
 * @brief 基于拉普拉斯核的孤立点检测
 *
 * @param src 输入图像；type: CV_8UC1
 * @param dst 输出图像
 * @return None
 */
void pointDetectLaplaceKernel(Mat &src, Mat &dst);

/**
 * @brief 基于拉普拉斯核的线检测
 *
 * @param src 输入图像；type: CV_8UC1
 * @param dst 输出图像
 * @param line_type 线类型；1 - 水平线, 2 - 西北东南方向线, 3 - 垂直线, 4 - 东北西南方向线
 * @return None
 */
void lineDetectLaplaceKernel(Mat &src, Mat &dst, int line_type);

// 边缘检测：
//  1. 降低噪声
//  2. 检测边缘，可参考 “spatial_filter.h” 中的高通部分
//      基本方法：计算图像的导数，即空间高通滤波，例如 Sobel 算子等；
//      进阶方法：在滤波的基础上增加了对图像噪声和边缘性质等因素的考虑，例如 Canny 算子等

/**
 * @brief 基于霍夫变换的线检测
 *
 * 调用 cv::HoughLines()
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @param rho 极坐标距离分辨率；"距离参数"为直线距离图像左上角点的距离，单位为像素
 * @param theta 极坐标角度分辨率；"角度参数"为直线与 x 轴的夹角角度，单位为弧度
 * @param threshold 累加器阈值参数，即直线筛选阈值
 * @param srn 精确累加器距离分辨率除数，即 accurate_rho = rho / srn
 * @param stn 精确累加器角度分辨率除数，即 accurate_theta = theta / stn
 * @param min_theta 最小角度
 * @param max_theta 最大角度
 * @return None
 */
void lineDetectHough(Mat &src, Mat &dst, double rho, double theta, int threshold, double srn = 0, double stn = 0,
                     double min_theta = 0, double max_theta = CV_PI);

/**
 * @brief 基于霍夫变换的线段检测
 *
 * 调用 cv::HoughLinesP()
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @param rho 极坐标距离分辨率；"距离参数"为直线距离图像左上角点的距离，单位为像素
 * @param theta 极坐标角度分辨率；"角度参数"为直线与 x 轴的夹角角度，单位为弧度
 * @param threshold 累加器阈值参数，即直线筛选阈值
 * @param minLineLength 线段最短长度
 * @param maxLineGap 线段最大两点间隔
 * @return None
 */
void lineSegmentDetectHough(Mat &src, Mat &dst, double rho, double theta, int threshold, double minLineLength = 0,
                            double maxLineGap = 0);

/**
 * @brief 基于霍夫变换的圆检测
 *
 * 调用 cv::HoughCircles()
 *
 * @param src 输入图像；type: CV_8UC1，灰度图
 * @param dst 输出图像
 * @param method 检测方法；见 cv::HoughModes，可选 #HOUGH_GRADIENT 或 #HOUGH_GRADIENT_ALT
 * @param dp 累加器图像分辨率与输入图像分辨率的反比；例如 dp=1 表示累加器与输入图像同分辨率，dp=2 表示宽高均为输入图像的一半
 * @param minDist 圆心之间的最小距离；过小则可能检测出相邻的多个圆，过大则可能漏检
 * @param param1 方法特定参数；在 #HOUGH_GRADIENT / #HOUGH_GRADIENT_ALT 中，它是传递给 Canny 边缘检测的高阈值（低阈值等于 param1 的一半）
 * @param param2 方法特定参数；在 #HOUGH_GRADIENT 中，它是累加器阈值，值越小则检测到的伪圆越多，值越大则只返回高置信度圆
 * @param minRadius 最小圆半径
 * @param maxRadius 最大圆半径；若 <= 0 则使用图像最大尺寸，若 < 0 则在 #HOUGH_GRADIENT 模式下仅返回圆心而不估计半径
 * @return None
 */
void circleDetectHough(Mat &src, Mat &dst, int method = HOUGH_GRADIENT, double dp = 1, double minDist = 20,
                       double param1 = 100, double param2 = 100, int minRadius = 0, int maxRadius = 0);

/**
 * @brief 基于 Harris 算法的角点检测
 *
 * 调用 cv::cornerHarris()
 *
 * @param src 输入图像；type: CV_8U 或 CV_32F
 * @param dst 输出图像
 * @param blockSize 邻域的大小；请参阅 #cornerEigenValsAndVecs
 * @param ksize Sobel 核尺寸；必须为正奇数
 * @param k 自由参数(经验常数)；建议取值范围为 (0.04, 0.06)
 * @param borderType 边界填充模式；默认为镜像填充，请参阅 #BorderTypes，不支持 #BORDER_WRAP
 * @return None
 */
void cornerDetectHarris(Mat &src, Mat &dst, int threshold, int blockSize, int ksize, double k = 0.04,
                        int borderType = BORDER_DEFAULT);

/**
 * @brief 基于 Shi-Tomasi 算法的角点检测
 *
 * 调用 cv::goodFeaturesToTrack()
 *
 * @param src 输入图像；type: CV_8U 或 CV_32F
 * @param dst 输出图像
 * @param maxCorners 返回的最大角点数
 * @param qualityLevel 最低质量阈值；角点特征值小于 "qualityLevel x 最大特征值" 的点将会被舍弃
 * @param minDistance 返回的角点之间的最小间距
 * @param mask 检测区域掩膜
 * @param blockSize 邻域的大小；请参阅 #cornerEigenValsAndVecs
 * @return None
 */
void cornerDetectShiTomasi(Mat &src, Mat &dst, int maxCorners, double qualityLevel, double minDistance,
                           InputArray mask = noArray(), int blockSize = 3);

/**
 * @brief 亚像素级角点检测
 *
 * 调用 cv::goodFeaturesToTrack(), cv::cornerSubPix()
 *
 * @param src 输入图像；type: CV_8U 或 CV_32F
 * @param dst 输出图像
 * @param maxCorners 返回的最大角点数
 * @param qualityLevel 最低质量阈值；角点特征值小于 "qualityLevel x 最大特征值" 的点将会被舍弃
 * @param minDistance 返回的角点之间的最小间距
 * @param winSize (亚像素级参数) 搜索窗口的半径
 * @param zeroZone (亚像素级参数) 搜索窗口中央死区的半径；用于避免自相关矩阵可能出现的奇点，值为 (-1, -1) 则表示没有死区
 * @param criteria (亚像素级参数) 角点迭代检测的终止条件；例如：`TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001)`
 * @param mask 检测区域掩膜
 * @param blockSize 邻域的大小；请参阅 #cornerEigenValsAndVecs
 * @param useHarrisDetector 是否使用 Harris 算法；默认为 false，即使用 Shi-Tomasi 算法
 * @param k Harris 算法的自由参数(经验常数)；建议取值范围为 (0.04, 0.06)
 * @return None
 */
void cornerDetectSubPixel(Mat &src, Mat &dst, int maxCorners, double qualityLevel, double minDistance, Size winSize,
                          Size zeroZone, TermCriteria criteria, InputArray mask = noArray(), int blockSize = 3,
                          bool useHarrisDetector = false, double k = 0.04);


#endif //DIP_OPENCV_IMAGE_SEGMENTATION_H
