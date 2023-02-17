#ifndef DIP_OPENCV_SPATIAL_FILTER_H
#define DIP_OPENCV_SPATIAL_FILTER_H

#include <iostream>
#include <stdexcept>
#include "opencv2/opencv.hpp"
#include "gray_transform.h"


using namespace std;
using namespace cv;


/**
 * @brief 线性空间滤波（即二维图像卷积）
 *
 * 调用 cv::filter2D()
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param kernel 卷积核
 * @return None
 */
void linearSpatialFilter(Mat &src, Mat &dst, Mat &kernel);

/**
 * @brief 盒式平滑（低通）空间滤波
 *
 * 调用 cv::boxFilter()
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param ksize 卷积核的尺寸
 * @param anchor 锚点；默认值为 Point(-1，-1)，表示锚点在内核中心
 * @param normalize 标志；指定内核是否按区域归一化
 * @param borderType 边界填充模式；默认为镜像填充，请参阅 #BorderTypes，不支持 #BORDER_WRAP
 * @return None
 */
void smoothSpatialFilterBox(Mat &src, Mat &dst, Size ksize, Point anchor = Point(-1, -1), bool normalize = true,
                            int borderType = BORDER_DEFAULT);

/**
 * @brief 高斯平滑（低通）空间滤波
 *
 * 调用 cv::GaussianBlur()
 *
 * @param src 输入图像；注意 depth 应为 CV_8U, CV_16U, CV_16S, CV_32F or CV_64F
 * @param dst 输出图像
 * @param ksize 高斯核的尺寸；ksize.width 和 ksize.height 可以不同，但均必须为正奇数，或者为零（再根据 sigma 进行计算）；
 *        注意卷积核的尺寸大于 (6 * sigma) 时无意义
 * @param sigmaX 高斯核在 X方向上的标准差
 * @param sigmaY 高斯核在 Y方向上的标准差；如果 sigmaY 为零，则设置为 sigmaX；如果两个 sigma 都为零，
 *        则分别根据 ksize.width 和 ksize.height 分别进行计算；有关细节请参阅 #getGaussianKernel；
 *        建议指定所有的参数，包括 ksize、sigmaX 和 sigmaY
 * @param borderType 边界填充模式；默认为镜像填充，请参阅 #BorderTypes，不支持 #BORDER_WRAP
 * @return None
 */
void smoothSpatialFilterGauss(Mat &src, Mat &dst, Size ksize, double sigmaX, double sigmaY = 0,
                              int borderType = BORDER_DEFAULT);

/**
 * @brief 阴影校正 TODO:
 *
 * @param src 输入图像；注意 depth 应为 CV_8U
 * @param dst 输出图像
 * @param k1 模糊处理比例系数；取值范围为 (0, 0.5]，系数越大，模糊程度越高
 * @return None
 */
void shadingCorrection(Mat &src, Mat &dst, float k1, float k2);

/**
 * @brief 统计排序（非线性）滤波器
 *
 * 中值滤波调用 cv::medianBlur()
 *
 * @param src 输入图像；注意当 ksize 为 3 or 5 时，depth 应为 CV_8U, CV_16U or CV_32F；
 *        而对于较大的孔径尺寸，depth 只能为 CV_8U
 * @param dst 输出图像
 * @param ksize 卷积核的尺寸；必须为正奇数
 * @param percentage 排序（从小到大）位置的百分比值；取值范围为 [0, 100]；默认为 50，即中值滤波，适用于去除椒盐噪声（冲激）
 * @return None
 */
void orderStatisticsFilter(Mat &src, Mat &dst, int ksize, int percentage = 50);

/**
 * @brief 拉普拉斯（二阶导数）锐化（高通）空间滤波
 *
 * 调用 cv::Laplacian()
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param ksize 高斯核的尺寸；必须为正奇数
 * @param scale 计算拉普拉斯值的比例因子；默认情况下无缩放，有关细节请参阅 #getDerivKernels
 * @param delta 偏移量
 * @param borderType 边界填充模式；默认为镜像填充，请参阅 #BorderTypes，不支持 #BORDER_WRAP
 * @return None
 */
void sharpenSpatialFilterLaplace(Mat &src, Mat &dst, int ksize = 1, double scale = 1, double delta = 0,
                                 int borderType = BORDER_DEFAULT);

/**
 * @brief 模板锐化，当 k = 1 时，其为钝化掩蔽；当 k > 1 时，其为高提升滤波；当 k < 1 时，可以减少钝化模板的贡献。
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param smooth_size 模糊（平滑）处理的卷积核尺寸
 * @param k 钝化模板权值；k >= 0；默认为 1，即钝化掩蔽
 * @return None
 */
void sharpenSpatialFilterTemplate(Mat &src, Mat &dst, Size smooth_size, float k = 1);

// Roberts 算子
void sharpenSpatialFilterRoberts();

// Prewitt 算子
void sharpenSpatialFilterPrewitt();

/**
 * @brief 索贝尔（一阶导数）锐化（高通）空间滤波
 *
 * 调用 cv::Sobel()
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param dx X 方向导数的阶
 * @param dy Y 方向导数的阶
 * @param ksize 高斯核的尺寸；必须为正奇数
 * @param scale 计算导数值的比例因子；默认情况下无缩放，有关细节请参阅 #getDerivKernels
 * @param delta 偏移量
 * @param borderType 边界填充模式；默认为镜像填充，请参阅 #BorderTypes，不支持 #BORDER_WRAP
 * @return None
 */
void sharpenSpatialFilterSobel(Mat &src, Mat &dst, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0,
                               int borderType = BORDER_DEFAULT);

/**
 * @brief 沙尔（一阶导数）锐化（高通）空间滤波，可等价于 ksize 为 3 的 Sobel
 *
 * 调用 cv::Scharr()
 *
 * @param src 输入图像
 * @param dst 输出图像
 * @param dx X 方向导数的阶
 * @param dy Y 方向导数的阶
 * @param scale 计算导数值的比例因子；默认情况下无缩放，有关细节请参阅 #getDerivKernels
 * @param delta 偏移量
 * @param borderType 边界填充模式；默认为镜像填充，请参阅 #BorderTypes，不支持 #BORDER_WRAP
 * @return None
 */
void sharpenSpatialFilterScharr(Mat &src, Mat &dst, int dx, int dy, double scale = 1, double delta = 0,
                               int borderType = BORDER_DEFAULT);

/**
 * @brief Canny 锐化（高通）空间滤波
 *
 * 调用 cv::Canny()
 *
 * @param src 输入图像；注意 depth 为 8-bit
 * @param dst 输出图像；单通道 8-bit 图像
 * @param threshold1 双阈值迟滞处理（第 4 步）的第一个阈值；minValue
 * @param threshold2 双阈值迟滞处理（第 4 步）的第二个阈值；maxValue
 * @param apertureSize Sobel's ksize
 * @param L2gradient 标志；指定是否应用更精确的方式计算图像梯度幅值
 * @return None
 */
void sharpenSpatialFilterCanny(Mat &src, Mat &dst, double threshold1, double threshold2, int apertureSize = 3,
                               bool L2gradient = false);


#endif //DIP_OPENCV_SPATIAL_FILTER_H
