#ifndef DIP_OPENCV_IMAGE_SEGMENTATION_H
#define DIP_OPENCV_IMAGE_SEGMENTATION_H

#include "common.h"
#include "gray_transform.h"
#include <opencv2/xfeatures2d.hpp>


using namespace cv::xfeatures2d;


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

/**
 * @brief 基于 SURF 算法的关键点特征检测
 *
 * 调用 cv::xfeatures2d::SURF
 *
 * @param src 输入图像；type: CV_8U 或 CV_32F
 * @param dst 输出图像
 * @param hessianThreshold 检测阈值
 * @param nOctaves 图像金字塔的层数
 * @param nOctaveLayers 图像金字塔的每一层的子层数
 * @param extended 扩展描述符标志；true - 生成的描述符具有 128 个元素，false - 生成的描述符具有 64 个元素
 * @param upright 直立或旋转特征标志；true - 不计算特征点的方向信息，false - 计算特征点的方向（即特征点能够适应图像的旋转变换）
 * @return None
 */
void keyPointDetectSurf(Mat &src, Mat &dst, double hessianThreshold = 100, int nOctaves = 4, int nOctaveLayers = 3,
                        bool extended = false, bool upright = false);


#endif //DIP_OPENCV_IMAGE_SEGMENTATION_H
