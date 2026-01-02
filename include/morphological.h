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
void grayToBinary(const cv::Mat &src, cv::Mat &dst, double thresh, double maxval, int type = cv::THRESH_BINARY);

/**
 * @brief 获取二值图像的最大值
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @return maxval
 */
uchar getBinaryMaxval(const cv::Mat &src);

/**
 * @brief 二值反转
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @return None
 */
void binaryInvert(const cv::Mat &src, cv::Mat &dst);

// 构建（形态学）结构元
// OpenCV cv::Mat cv::getStructuringElement(int shape, cv::Size ksize, cv::Point anchor = cv::Point(-1,-1));

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
void morphologyErode(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);

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
void morphologyDilate(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);

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
void morphologyOpen(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);

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
void morphologyClose(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);

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
void morphologyHMT(const cv::Mat &src, cv::Mat &dst, const cv::Mat &fore_kernel, const cv::Mat &back_kernel);

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
void morphologyGradient(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);

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
void morphologyTophat(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);

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
void morphologyBlackhat(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);

/**
 * @brief 边界提取
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @param size 结构元尺寸；通常取决于所需的边界尺寸
 * @return None
 */
void boundaryExtract(const cv::Mat &src, cv::Mat &dst, int size);

/**
 * @brief 孔洞填充
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @param start 孔洞起始点图像；type: CV_8UC1，二值图
 * @return None
 */
void holeFill(const cv::Mat &src, cv::Mat &dst, const cv::Mat &start);

/**
 * @brief 提取连通分量
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @return Number of connected
 */
int extractConnected(const cv::Mat &src, cv::Mat &dst);

// 连通域提取
// OpenCV int cv::connectedComponents(cv::InputArray image, cv::OutputArray labels, int connectivity, int ltype, int ccltype);

// 连通域提取（详细）
// OpenCV int cv::connectedComponentsWithStats(cv::InputArray image, cv::OutputArray labels, cv::OutputArray stats,
//                                             cv::OutputArray centroids, int connectivity, int ltype, int ccltype);

// 凸壳、细化、粗化、骨架、裁剪

/**
 * @brief 腐蚀形态学重建
 *
 * @param src 输入图像（标记图像）；type: CV_8UC1，二值图或灰度图
 * @param tmpl 模板图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @return None
 */
void erodeReconstruct(const cv::Mat &src, const cv::Mat &tmpl, cv::Mat &dst);

/**
 * @brief 膨胀形态学重建
 *
 * @param src 输入图像（标记图像）；type: CV_8UC1，二值图或灰度图
 * @param tmpl 模板图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @return None
 */
void dilateReconstruct(const cv::Mat &src, const cv::Mat &tmpl, cv::Mat &dst);

/**
 * @brief 开运算形态学重建
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param erode_kernel 腐蚀结构元；可以使用 #getStructuringElement 创建
 * @param erode_times 腐蚀次数
 * @return None
 */
void openReconstruct(const cv::Mat &src, cv::Mat &dst, const cv::Mat &erode_kernel, int erode_times = 1);

/**
 * @brief 闭运算形态学重建
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param dilate_kernel 膨胀结构元；可以使用 #getStructuringElement 创建
 * @param dilate_times 膨胀次数
 * @return None
 */
void closeReconstruct(const cv::Mat &src, cv::Mat &dst, const cv::Mat &dilate_kernel, int dilate_times = 1);

/**
 * @brief 顶帽形态学重建
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param erode_kernel 腐蚀结构元；可以使用 #getStructuringElement 创建
 * @param erode_times 腐蚀次数
 * @return None
 */
void tophatReconstruct(const cv::Mat &src, cv::Mat &dst, const cv::Mat &erode_kernel, int erode_times = 1);

/**
 * @brief 底帽形态学重建
 *
 * @param src 输入图像；type: CV_8UC1，二值图或灰度图
 * @param dst 输出图像
 * @param dilate_kernel 膨胀结构元；可以使用 #getStructuringElement 创建
 * @param dilate_times 膨胀次数
 * @return None
 */
void blackhatReconstruct(const cv::Mat &src, cv::Mat &dst, const cv::Mat &dilate_kernel, int dilate_times = 1);

/**
 * @brief 孔洞填充（自动版）
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @return None
 */
void holeFill(const cv::Mat &src, cv::Mat &dst);

/**
 * @brief 边界清除
 *
 * @param src 输入图像；type: CV_8UC1，二值图
 * @param dst 输出图像
 * @return None
 */
void borderClear(const cv::Mat &src, cv::Mat &dst);


#endif //DIP_OPENCV_MORPHOLOGICAL_H
