#ifndef DIP_OPENCV_MORPHOLOGICAL_H
#define DIP_OPENCV_MORPHOLOGICAL_H

#include "common.h"


/**
 * @brief GRAY 转换为 Binary (二值化)
 *
 * 调用 cv::threshold()
 *
 * @param src 输入图像；type: CV_8UC1，灰度图
 * @param dst 输出图像
 * @param thresh 二值图像的阈值
 * @param maxval 二值图像的最大值
 * @param type 二值化操作类型，请参阅 #ThresholdTypes
 * @return None
 */
void grayToBinary(Mat &src, Mat &dst, double thresh, double maxval, int type = THRESH_BINARY);

/**
 * @brief 获取二值图像的最大值
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @return maxval
 */
uchar getBinaryMaxval(Mat &src);

/**
 * @brief 二值反转
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @return None
 */
void binaryInvert(Mat &src, Mat &dst);

// 构建（形态学）结构元
// OpenCV Mat getStructuringElement(int shape, Size ksize, Point anchor = Point(-1,-1));

/**
 * @brief 形态学腐蚀
 *
 * 调用 cv::erode() / cv::morphologyEx(cv::MORPH_ERODE)
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param kernel 结构元；可以使用 #getStructuringElement 创建
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
 * @param kernel 结构元；可以使用 #getStructuringElement 创建
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
 * @param kernel 结构元；可以使用 #getStructuringElement 创建
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
 * @param kernel 结构元；可以使用 #getStructuringElement 创建
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
 * @param fore_kernel 前景结构元（击中）；可以使用 #getStructuringElement 创建
 * @param back_kernel 背景结构元（击不中）；可以使用 #getStructuringElement 创建
 * @return None
 */
void morphologyHMT(Mat &src, Mat &dst, const Mat &fore_kernel, const Mat &back_kernel);

/**
 * @brief 形态学梯度
 *
 * 调用 cv::morphologyEx(cv::MORPH_GRADIENT)
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param kernel 结构元；可以使用 #getStructuringElement 创建
 * @return None
 */
void morphologyGradient(Mat &src, Mat &dst, const Mat &kernel);

/**
 * @brief 形态学顶帽变换
 *
 * 调用 cv::morphologyEx(cv::MORPH_TOPHAT)
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param kernel 结构元；可以使用 #getStructuringElement 创建
 * @return None
 */
void morphologyTophat(Mat &src, Mat &dst, const Mat &kernel);

/**
 * @brief 形态学底帽变换
 *
 * 调用 cv::morphologyEx(cv::MORPH_BLACKHAT)
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param kernel 结构元；可以使用 #getStructuringElement 创建
 * @return None
 */
void morphologyBlackhat(Mat &src, Mat &dst, const Mat &kernel);

/**
 * @brief 边界提取
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @param size 结构元尺寸；通常取决于所需的边界尺寸
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
 * @return Number of connected
 */
int extractConnected(Mat &src, Mat &dst);

// 连通域提取
// OpenCV int connectedComponents(InputArray image, OutputArray labels, int connectivity, int ltype, int ccltype);

// 连通域提取（详细）
// OpenCV int connectedComponentsWithStats(InputArray image, OutputArray labels, OutputArray stats,
//                                         OutputArray centroids, int connectivity, int ltype, int ccltype);

// 凸壳、细化、粗化、骨架、裁剪

/**
 * @brief 腐蚀形态学重建
 *
 * @param src 输入图像（标记图像）；type: CV_8UC1，二值图或灰度图
 * @param tmpl 模板图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @return None
 */
void erodeReconstruct(Mat &src, const Mat &tmpl, Mat &dst);

/**
 * @brief 膨胀形态学重建
 *
 * @param src 输入图像（标记图像）；type: CV_8UC1，二值图或灰度图
 * @param tmpl 模板图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @return None
 */
void dilateReconstruct(Mat &src, const Mat &tmpl, Mat &dst);

/**
 * @brief 开运算形态学重建
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param erode_kernel 腐蚀结构元；可以使用 #getStructuringElement 创建
 * @param erode_times 腐蚀次数
 * @return None
 */
void openReconstruct(Mat &src, Mat &dst, const Mat &erode_kernel, int erode_times = 1);

/**
 * @brief 闭运算形态学重建
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param dilate_kernel 膨胀结构元；可以使用 #getStructuringElement 创建
 * @param dilate_times 膨胀次数
 * @return None
 */
void closeReconstruct(Mat &src, Mat &dst, const Mat &dilate_kernel, int dilate_times = 1);

/**
 * @brief 顶帽形态学重建
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param erode_kernel 腐蚀结构元；可以使用 #getStructuringElement 创建
 * @param erode_times 腐蚀次数
 * @return None
 */
void tophatReconstruct(Mat &src, Mat &dst, const Mat &erode_kernel, int erode_times = 1);

/**
 * @brief 底帽形态学重建
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param dilate_kernel 膨胀结构元；可以使用 #getStructuringElement 创建
 * @param dilate_times 膨胀次数
 * @return None
 */
void blackhatReconstruct(Mat &src, Mat &dst, const Mat &dilate_kernel, int dilate_times = 1);

/**
 * @brief 孔洞填充（自动版）
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @return None
 */
void holeFill(Mat &src, Mat &dst);

/**
 * @brief 边界清除
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @return None
 */
void borderClear(Mat &src, Mat &dst);


#endif //DIP_OPENCV_MORPHOLOGICAL_H
