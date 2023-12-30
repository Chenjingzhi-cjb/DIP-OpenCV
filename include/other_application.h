#ifndef DIP_OPENCV_OTHER_APPLICATION_H
#define DIP_OPENCV_OTHER_APPLICATION_H

#include "common.h"


/**
 * @brief 计算图像偏移
 *
 * @param image_tmpl 模板图像
 * @param image_offset 偏移图像
 * @return The offset of the image_tmpl in the image_offset (the result is in the central coordinate system)
 */
pair<double, double> calcImageOffset(Mat &image_tmpl, Mat &image_offset);

/**
 * @brief 计算图像偏移
 *
 * @param image_std 标准图像
 * @param image_offset 偏移图像
 * @param tmpl_divisor 模板除数，中心模板截取尺寸 = 标准图像尺寸 / 模板除数，其值必须 > 1
 * @return The offset of image_offset relative to image_std (the result is in the central coordinate system)
 */
pair<double, double> calcImageOffset(Mat &image_std, Mat &image_offset, double tmpl_divisor);

/**
 * @brief 计算图像清晰度值
 *
 * @param image 输入图像
 * @return The sharpness score of image
 */
double calcImageSharpness(Mat &image);


#endif //DIP_OPENCV_OTHER_APPLICATION_H
