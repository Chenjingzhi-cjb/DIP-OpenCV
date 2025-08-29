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
 * @return confidence
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

enum class CornerDescriptorType {
    ORB,
    SIFT
};

/**
 * @brief 检测图像的角点并计算特征点与描述子
 *        使用 goodFeaturesToTrack + cornerSubPix + ORB / SIFT
 *
 * @param image 输入图像
 * @param key_points 计算结果 - 特征点
 * @param descriptors 计算结果 - 描述子
 * @param descriptor_type 描述子算法类型
 * @param maxCorners 最大角点数量
 * @param qualityLevel 角点质量阈值
 * @param minDistance 最小角点距离
 * @param winSize 角点亚像素优化窗口大小
 * @param criteria 角点亚像素优化迭代终止条件
 * @param min_Corners 最小角点数量
 * @param keypoint_diameter 特征点直径
 * @return None
 */
void cornerDetectAndCompute(const cv::Mat &image, std::vector<cv::KeyPoint> &key_points, cv::Mat &descriptors,
                            CornerDescriptorType descriptor_type = CornerDescriptorType::ORB,
                            int maxCorners = 200, double qualityLevel = 0.01, double minDistance = 10,
                            cv::Size winSize = cv::Size(11, 11),
                            cv::TermCriteria criteria = cv::TermCriteria(
                                    cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1),
                            int min_Corners = 8, float keypoint_diameter = 7.0f);

/**
 * @brief 基于 Brute-Force 匹配器计算并筛选优质匹配组
 *
 * @param key_points_tmpl 模板图像特征点
 * @param key_points_dst 目标图像特征点
 * @param descriptors_tmpl 模板图像描述子
 * @param descriptors_dst 目标图像描述子
 * @param good_matches 计算结果 - 匹配组
 * @return Status
 */
bool calcGoodMatchesByBFMatcher(std::vector<cv::KeyPoint> &key_points_tmpl, std::vector<cv::KeyPoint> &key_points_dst,
                                cv::Mat &descriptors_tmpl, cv::Mat &descriptors_dst,
                                std::vector<cv::DMatch> &good_matches);

/**
 * @brief 基于 FLANN 匹配器计算并筛选优质匹配组
 *
 * @param key_points_tmpl 模板图像特征点
 * @param key_points_dst 目标图像特征点
 * @param descriptors_tmpl 模板图像描述子
 * @param descriptors_dst 目标图像描述子
 * @param good_matches 计算结果 - 匹配组
 * @return Status
 */
bool
calcGoodMatchesByFLANNMatcher(std::vector<cv::KeyPoint> &key_points_tmpl, std::vector<cv::KeyPoint> &key_points_dst,
                              cv::Mat &descriptors_tmpl, cv::Mat &descriptors_dst,
                              std::vector<cv::DMatch> &good_matches);

/**
 * @brief 计算模板（基于特征）在目标图像中的位置（中心坐标系）
 *        使用 ORB
 *
 * @param image_tmpl 模板图像
 * @param image_dst 目标图像
 * @param position 计算结果 - 位置坐标
 * @param scale 计算结果 - 缩放比例（可为 nullptr）
 * @param angle 计算结果 - 旋转角度（可为 nullptr）
 * @param pre_features_num 预检测的特征点数量
 * @return confidence
 */
double
calcTemplatePositionORB(const cv::Mat &image_tmpl, const cv::Mat &image_dst, cv::Point2d &position, double *scale,
                        double *angle, int pre_features_num = 500);

/**
 * @brief 计算模板（基于特征）在目标图像中的位置（中心坐标系）
 *        使用 goodFeaturesToTrack + cornerSubPix + ORB / SIFT
 *
 * @param image_tmpl 模板图像; type: CV_8UC1 or CV_32FC1
 * @param image_dst 目标图像; type: 与 image_tmpl 保持一致
 * @param position 计算结果 - 位置坐标
 * @param scale 计算结果 - 缩放比例（可为 nullptr）
 * @param angle 计算结果 - 旋转角度（可为 nullptr）
 * @param descriptor_type 描述子算法类型
 * @return confidence
 */
double
calcTemplatePositionCorner(const cv::Mat &image_tmpl, const cv::Mat &image_dst, cv::Point2d &position, double *scale,
                           double *angle, CornerDescriptorType descriptor_type = CornerDescriptorType::ORB);

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
