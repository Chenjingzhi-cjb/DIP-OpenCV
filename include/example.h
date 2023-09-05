#ifndef DIP_OPENCV_EXAMPLE_H
#define DIP_OPENCV_EXAMPLE_H

#include "common.h"
#include "gray_transform.h"
#include "frequency_filter.h"
#include "wavelet_transform.h"


using namespace std;
using namespace cv;


// 3. 灰度变换与空间滤波 灰度变换 gray_transform.h

/**
 * @brief 阴影校正示例
 *
 * @return None
 */
void shadingCorrectionExample();


// 4. 频率域滤波 frequency_filter.h

/**
 * @brief 空间域图像与频率域图像的转换示例
 *
 * @return None
 */
void domainTransformExample();


// 7. 小波变换和其他图像变换 wavelet_transform.h

/**
 * @brief 离散余弦变换及逆变换示例
 *
 * @return None
 */
void DCTExample();

/**
 * @brief 分块离散余弦变换及逆变换示例
 *
 * @return None
 */
void blockDCTExample();


#endif //DIP_OPENCV_EXAMPLE_H
