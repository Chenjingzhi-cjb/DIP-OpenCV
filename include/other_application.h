#ifndef DIP_OPENCV_OTHER_APPLICATION_H
#define DIP_OPENCV_OTHER_APPLICATION_H

#include "common.h"


/**
 * @brief 计算模板在目标图像中的位置（中心坐标系）
 *
 * @param image_tmpl 模板图像; type: CV_8UC1 or CV_8UC3 or CV_32FC1 or CV_32FC3
 * @param image_dst 目标图像; type: 与 image_tmpl 保持一致
 * @param position 计算结果 - 位置坐标
 * @param sub_pixel 是否使用亚像素级算法
 * @return match score
 */
double calcTemplatePosition(const cv::Mat &image_tmpl, const cv::Mat &image_dst, cv::Point2d &position,
                            bool sub_pixel = false);

/**
 * @brief 计算目标图像相对于原图像的位移（中心坐标系）
 *
 * @param image_src 原图像; type: CV_32FC1 or CV_64FC1
 * @param image_dst 目标图像; type: 与 image_src 保持一致
 * @param offset 计算结果 - 位移量
 * @return confidence
 */
double calcImageOffset(const cv::Mat &image_src, const cv::Mat &image_dst, cv::Point2d &offset);

/**
 * @brief 计算图像清晰度值
 *
 * @param image 输入图像
 * @return The sharpness score of image
 */
double calcImageSharpness(Mat &image);

/**
 * @brief 计算图像清晰度值（优化版）
 *
 * @param image 输入图像
 * @param part_count 分区数量，其值应为 >2 的偶数
 * @return The sharpness score of image
 */
double calcSharpnessOldOpt(cv::Mat *image, int part_count);


#endif //DIP_OPENCV_OTHER_APPLICATION_H
