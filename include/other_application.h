#ifndef DIP_OPENCV_OTHER_APPLICATION_H
#define DIP_OPENCV_OTHER_APPLICATION_H

#include "common.h"


/**
 * @brief 计算图像偏移
 *
 * @param image_std 标准图像
 * @param image_offset 偏移图像，尺寸必须与标准图像相同
 * @param tmpl_divisor 模板除数，中心模板截取尺寸 = 标准图像尺寸 / 模板除数，其值必须 > 1
 * @return The offset of image_offset relative to image_std (the result is in the central coordinate system)
 */
pair<double, double> calcImageOffset(Mat &image_std, Mat &image_offset, double tmpl_divisor = 3);


#endif //DIP_OPENCV_OTHER_APPLICATION_H
