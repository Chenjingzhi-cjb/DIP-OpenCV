#ifndef DIP_OPENCV_EXAMPLE_H
#define DIP_OPENCV_EXAMPLE_H

#include "gray_transform.h"
#include "frequency_filter.h"
#include "color_process.h"
#include "morphological.h"
#include "wavelet_transform.h"


// 3. 灰度变换与空间滤波 灰度变换 gray_transform.h

/**
 * @brief 局部直方图均衡化示例
 *
 * @return None
 */
void localEqualizeHistExample();

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

/**
 * @brief 高频增强滤波示例
 *
 * @return None
 */
void highFreqEmphasisExample();


// 6. 彩色图像处理

/**
 * @brief HSI 转换示例
 *
 * @return None
 */
void hsiExample();


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


// 9. 形态学图像处理

/**
 * @brief 孔洞填充示例
 *
 * @return None
 */
void holeFillExample();

/**
 * @brief 边界清除示例
 *
 * @return None
 */
void borderClearExample();

/**
 * @brief 使用灰度级形态学重建展平复杂背景示例
 *
 * @return None
 */
void morphFlattenBackgroundExample();  // TODO: 使用 msvc 编译器编译后，运行时出现未知原因导致的延迟


#endif //DIP_OPENCV_EXAMPLE_H
