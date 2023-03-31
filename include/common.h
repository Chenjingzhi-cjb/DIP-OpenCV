#ifndef DIP_OPENCV_COMMON_H
#define DIP_OPENCV_COMMON_H

#include <iostream>
#include <stdexcept>
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;


/**
 * @brief 打印图像数据
 *
 * @param image 图像
 * @param shrink_size 图像缩小的尺寸
 * @param preview_unit 图像预览的单位元素个数
 * @return None
 */
void printImageData(Mat &image, Size shrink_size = Size(64, 64), int preview_unit = 3);


#endif //DIP_OPENCV_COMMON_H
