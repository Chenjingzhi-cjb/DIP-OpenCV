#ifndef DIP_OPENCV_FEATURE_DETECTION_H
#define DIP_OPENCV_FEATURE_DETECTION_H

#include "common.h"
#include "gray_transform.h"


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


#endif //DIP_OPENCV_FEATURE_DETECTION_H
