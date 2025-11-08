#ifndef DIP_OPENCV_COMMON_H
#define DIP_OPENCV_COMMON_H

#include <iostream>
#include <stdexcept>
#include "opencv2/opencv.hpp"
#include <cmath>
#include <utility>
#include <vector>
#include <algorithm>
#include <numeric>

#include "FileEnumerator.hpp"
#include "Stopwatch.hpp"
#include "logger.hpp"


/**
 * @brief 打印图像数据
 *
 * @param image 图像
 * @param shrink_size 图像缩小的尺寸
 * @param preview_unit 图像预览的单位元素个数
 * @return None
 */
void printImageData(const cv::Mat &image, cv::Size shrink_size = cv::Size(64, 64), int preview_unit = 3);

/**
 * @brief 视频图像遍历
 *
 * @param video 视频对象
 * @return None
 */
void videoTraverse(cv::VideoCapture &video);


#endif //DIP_OPENCV_COMMON_H
