# DIP-OpenCV: 数字图像处理 OpenCV_Cpp

本项目基于冈萨雷斯的《数字图像处理》(第四版)，使用了 OpenCV_Cpp 对该书的一些理论知识进行了代码实现。



## 内容介绍（按章节）

### 3. 灰度变换与空间滤波

- 灰度变换 **gray_transform.h**

```cpp
void bgrToGray(const cv::Mat &src, cv::Mat &dst);  // BGR 转换为 GRAY

void grayLinearScaleCV_8U(const cv::Mat &src, cv::Mat &dst);  // 灰度线性缩放，缩放至 [0-255]

void grayInvert(const cv::Mat &src, cv::Mat &dst);  // 灰度反转（属于灰度线性变换）

void grayLog(const cv::Mat &src, cv::Mat &dst);  // 灰度对数变换

void grayAntiLog(const cv::Mat &src, cv::Mat &dst);  // 灰度反对数变换

void grayGamma(const cv::Mat &src, cv::Mat &dst, float gamma);  // 灰度伽马变换，也称幂律变换

void grayContrastStretch(const cv::Mat &src, cv::Mat &dst, uint r1, uint s1, uint r2, uint s2);  // 灰度对比度拉伸

void grayLayering(const cv::Mat &src, cv::Mat &dst, uint r1, uint r2, uint s, bool other_zero);  // 灰度值级分层

void grayBitPlaneLayering(const cv::Mat &src, std::vector<cv::Mat> &dst);  // 灰度比特平面分层

cv::Mat grayHistogram(const cv::Mat &src, const cv::Mat &mask = cv::Mat(), cv::Size size = cv::Size(512, 400), const cv::Scalar &color = cv::Scalar(255, 255, 255));  // 灰度直方图，即单通道直方图

// OpenCV void cv::equalizeHist( cv::InputArray src, cv::OutputArray dst );  全局直方图均衡化

void localEqualizeHist(const cv::Mat &src, cv::Mat &dst, double clipLimit = 40.0, cv::Size tileGridSize = cv::Size(8, 8));  // 局部直方图均衡化

void matchHist(const cv::Mat &src, cv::Mat &dst, cv::Mat &refer);  // 直方图规定化

void shadingCorrection(const cv::Mat &src, cv::Mat &dst, float k1 = 0.25, float k2 = 6);  // 阴影校正
```

- 空间滤波 **spatial_filter.h**

```cpp
void linearSpatialFilter(const cv::Mat &src, cv::Mat &dst, cv::Mat &kernel);  // 线性空间滤波（即二维图像卷积）

void smoothSpatialFilterBox(const cv::Mat &src, cv::Mat &dst, cv::Size ksize, cv::Point anchor = cv::Point(-1, -1), bool normalize = true, int borderType = cv::BORDER_DEFAULT);  // 盒式平滑（低通）空间滤波

void smoothSpatialFilterGauss(const cv::Mat &src, cv::Mat &dst, cv::Size ksize, double sigmaX, double sigmaY = 0, int borderType = cv::BORDER_DEFAULT);  // 高斯平滑（低通）空间滤波

void orderStatisticsFilter(const cv::Mat &src, cv::Mat &dst, int ksize, int percentage = 50);  // 统计排序（非线性）滤波器

void sharpenSpatialFilterLaplace(const cv::Mat &src, cv::Mat &dst, int ksize = 1, double scale = 1, double delta = 0, int borderType = cv::BORDER_DEFAULT);  // 拉普拉斯（二阶导数）锐化（高通）空间滤波

void sharpenSpatialFilterTemplate(const cv::Mat &src, cv::Mat &dst, cv::Size smooth_size, float k = 1);  // 模板锐化（钝化掩蔽、高提升滤波）

void sharpenSpatialFilterRoberts();  // Roberts 算子 TODO:

void sharpenSpatialFilterPrewitt();  // Prewitt 算子 TODO:

void sharpenSpatialFilterSobel(const cv::Mat &src, cv::Mat &dst, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = cv::BORDER_DEFAULT);  // 索贝尔（一阶导数）锐化（高通）空间滤波

void sharpenSpatialFilterScharr(const cv::Mat &src, cv::Mat &dst, int dx, int dy, double scale = 1, double delta = 0, int borderType = cv::BORDER_DEFAULT);  // 沙尔（一阶导数）锐化（高通）空间滤波

void sharpenSpatialFilterCanny(const cv::Mat &src, cv::Mat &dst, double threshold1, double threshold2, int apertureSize = 3, bool L2gradient = false);  // Canny 锐化（高通）空间滤波（边缘检测优化算法）
```

- **example.h**

```cpp
void localEqualizeHistExample();  // 局部直方图均衡化示例

void shadingCorrectionExample();  // 阴影校正示例
```

### 4. 频率域滤波

- **frequency_filter.h**

```cpp
void dftShift(cv::Mat &image);  // 傅里叶图像象限变换

void spatialToFrequency(const cv::Mat &src, cv::Mat &dst_complex);  // 图像空间域转频率域

void splitFrequencyMagnitude(const cv::Mat &src_complex, cv::Mat &dst_magnitude);  // 从频率域复数图像中分离出频率域实部幅值图像

void frequencyToSpatial(const cv::Mat &src_complex, cv::Mat &dst, const cv::Size &original_size);  // 图像频率域转空间域

cv::Mat idealLowPassFreqKernel(const cv::Size &size, int sigma);  // 理想低通频率滤波核函数，该核有振铃效应

cv::Mat gaussLowPassFreqKernel(const cv::Size &size, int sigma);  // 高斯低通频率滤波核函数

cv::Mat bwLowPassFreqKernel(const cv::Size &size, int sigma, int order);  // 巴特沃斯低通频率滤波核函数

cv::Mat idealHighPassFreqKernel(const cv::Size &size, int sigma);  // 理想高通频率滤波核函数，该核有振铃效应

cv::Mat gaussHighPassFreqKernel(const cv::Size &size, int sigma);  // 高斯高通频率滤波核函数

cv::Mat bwHighPassFreqKernel(const cv::Size &size, int sigma, int order);  // 巴特沃斯高通频率滤波核函数

cv::Mat highFreqEmphasisKernel(const cv::Size &size, int sigma, float k1 = 1, float k2 = 1);  // 高频增强滤波核函数

cv::Mat homomorphicEmphasisKernel(const cv::Size &size, int sigma, float gamma_h, float gamma_l, int c);  // 同态增强滤波核函数

cv::Mat idealBandRejectFreqKernel(const cv::Size &size, int C0, int width);  // 理想带阻频率滤波核函数

cv::Mat gaussBandRejectFreqKernel(const cv::Size &size, int C0, int width);  // 高斯带阻频率滤波核函数

cv::Mat bwBandRejectFreqKernel(const cv::Size &size, int C0, int width, int order);  // 巴特沃斯带阻频率滤波核函数

// cv::Mat notchBandRejectFreqKernel();  陷波带阻滤波核（定制化，主要用于处理周期噪声）

void frequencyFilter(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel, bool rm_negative = false);  // 频率域滤波

cv::Mat laplaceFreqKernel(const cv::Size &size);  // 拉普拉斯频率滤波核函数

void freqSharpenLaplace(const cv::Mat &src, cv::Mat &dst);  // 拉普拉斯频率域锐化

void frequencyFilterPlMul(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel, bool rm_negative = false);  // 频率域滤波（复数乘法版）
```

- **example.h**

```cpp
void domainTransformExample();  // 空间域图像与频率域图像的转换示例

void highFreqEmphasisExample();  // 高频增强滤波示例
```

### 5. 图像复原与重构

- **image_noise.h**

```cpp
void addNoiseGauss(const cv::Mat &src, cv::Mat &dst, int mean, int sigma);  // 添加高斯噪声

void addNoiseMean(const cv::Mat &src, cv::Mat &dst, int lower, int upper);  // 添加平均噪声

void addNoiseRayleigh(const cv::Mat &src, cv::Mat &dst, double sigma);  // 添加瑞利噪声

void addNoiseGamma(const cv::Mat &src, cv::Mat &dst, double sigma, double alpha, double beta);  // 添加伽马(爱尔兰）噪声

void addNoiseExp(const cv::Mat &src, cv::Mat &dst, double lambda);  // 添加指数噪声

void addNoiseSaltPepper(const cv::Mat &src, cv::Mat &dst, double noise_level, int type = 0, double salt_value = 255, double pepper_value = 0);  // 添加椒盐(冲激)噪声
```

- **spatial_filter.h**

```cpp
void geometricMeanFilter(const cv::Mat &src, cv::Mat &dst, cv::Size ksize);  // 几何均值滤波器，效果优于算术平均滤波器（即盒式滤波器）

void harmonicAvgFilter(const cv::Mat &src, cv::Mat &dst, cv::Size ksize);  // 谐波平均滤波器，能够处理 盐粒噪声 或 类高斯噪声，不能处理 胡椒噪声

void antiHarmonicAvgFilter(const cv::Mat &src, cv::Mat &dst, cv::Size ksize, float order);  // 反谐波平均滤波器，能够处理 盐粒噪声 或 胡椒噪声 或 类高斯噪声

void midPointFilter(const cv::Mat &src, cv::Mat &dst, cv::Size ksize);  // 中点滤波器，适合处理随机分布的噪声，如 高斯噪声 或 均匀噪声

void modifiedAlphaMeanFilter(const cv::Mat &src, cv::Mat &dst, cv::Size ksize, int d);  // 修正阿尔法滤波器，适合处理多种混合噪声

void adaptiveLocalFilter(const cv::Mat &src, cv::Mat &dst, cv::Size ksize);  // 自适应局部降噪滤波器

void adaptiveMedianFilter(const cv::Mat &src, cv::Mat &dst, int max_ksize);  // 自适应中值滤波器，能够去除椒盐噪声、平滑其他非冲激噪声且减少失真
```

- **frequency_filter.h**

```cpp
void bestNotchFilter(const cv::Mat &src, cv::Mat &dst, const cv::Mat &nbp_kernel, const cv::Size &opt_ksize);  // 最优陷波滤波
```

### 6. 彩色图像处理

- **color_process.h**

```cpp
std::vector<cv::Mat> colorChannelSpilt(const cv::Mat &src);  // 彩色通道分离

cv::Mat colorChannelMerge(const std::vector<cv::Mat> &channels);  // 彩色通道合并

void bgrToHsi(const cv::Mat &src, cv::Mat &dst);  // BGR 转换为 HSI

void hsiToBgr(const cv::Mat &src, cv::Mat &dst);  // HSI 转换为 BGR

void pseudoColor(const cv::Mat &src, cv::Mat &dst, cv::ColormapTypes color = cv::COLORMAP_JET);  // 伪彩色处理

void complementaryColor(const cv::Mat &src, cv::Mat &dst);  // 补色处理，即彩色反转

void colorLayering(const cv::Mat &src, cv::Mat &dst, const cv::Vec3b& color_center, double range_radius = 120);  // 彩色分层

// 彩色图像（RGB / HSI）的校正（对数变换 / 反对数变换 / 伽马变换）：参考灰度

void colorEqualizeHist(const cv::Mat &src, cv::Mat &dst);  // 彩色全局直方图均衡化（不建议使用）
```

- **example.h**

```cpp
void hsiExample();  // HSI 转换示例
```

### 7. 小波变换和其他图像变换

- **wavelet_transform.h**

```cpp
void DWT(const cv::Mat &src, cv::Mat &dst, const std::string &wname, int level);  // 离散小波变换

void IDWT(const cv::Mat &src, cv::Mat &dst, const std::string &wname, int level);  // 离散小波逆变换

void DCT(const cv::Mat &src, cv::Mat &dst);  // 离散余弦变换

void IDCT(const cv::Mat &src, cv::Mat &dst, const cv::Size &original_size);  // 离散余弦逆变换

void blockDCT(const cv::Mat &src, cv::Mat &dst, int block_size = 8);  // 分块离散余弦变换

void blockIDCT(const cv::Mat &src, cv::Mat &dst, const cv::Size &original_size, int block_size = 8);  // 分块离散余弦逆变换
```

- **example.h**

```cpp
void DCTExample();  // 离散余弦变换及逆变换示例

void blockDCTExample();  // 分块离散余弦变换及逆变换示例
```

### 9. 形态学图像处理

- **morphological.h**

```cpp
void grayToBinary(const cv::Mat &src, cv::Mat &dst, double thresh, double maxval, int type);  // GRAY 转换为 Binary (二值化)

uchar getBinaryMaxval(const cv::Mat &src);  // 获取二值图像的最大值

void binaryInvert(const cv::Mat &src, cv::Mat &dst);  // 二值反转

// OpenCV cv::Mat getStructuringElement(int shape, cv::Size ksize, cv::Point anchor = cv::Point(-1,-1));  构建（形态学）结构元

void morphologyErode(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);  // 形态学腐蚀

void morphologyDilate(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);  // 形态学膨胀

void morphologyOpen(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);  // 形态学开运算

void morphologyClose(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);  // 形态学闭运算

void morphologyHMT(const cv::Mat &src, cv::Mat &dst, const cv::Mat &fore_kernel, const cv::Mat &back_kernel);  // 形态学击中击不中变换

void morphologyGradient(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);  // 形态学梯度

void morphologyTophat(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);  // 形态学顶帽变换

void morphologyBlackhat(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel);  // 形态学底帽变换

void boundaryExtract(const cv::Mat &src, cv::Mat &dst, int size);  // 边界提取

void holeFill(const cv::Mat &src, cv::Mat &dst, const cv::Mat &start);  // 孔洞填充

void extractConnected(const cv::Mat &src, cv::Mat &dst);  // 提取连通分量

// 凸壳、细化、粗化、骨架、裁剪

void erodeReconstruct(const cv::Mat &src, const cv::Mat &tmpl, cv::Mat &dst);  // 腐蚀形态学重建

void dilateReconstruct(const cv::Mat &src, const cv::Mat &tmpl, cv::Mat &dst);  // 膨胀形态学重建

void openReconstruct(const cv::Mat &src, cv::Mat &dst, const cv::Mat &erode_kernel, int erode_times = 1);  // 开运算形态学重建

void closeReconstruct(const cv::Mat &src, cv::Mat &dst, const cv::Mat &dilate_kernel, int dilate_times = 1);  // 闭运算形态学重建

void tophatReconstruct(const cv::Mat &src, cv::Mat &dst, const cv::Mat &erode_kernel, int erode_times = 1);  // 顶帽形态学重建

void blackhatReconstruct(const cv::Mat &src, cv::Mat &dst, const cv::Mat &dilate_kernel, int dilate_times = 1);  // 底帽形态学重建

void holeFill(const cv::Mat &src, cv::Mat &dst);  // 孔洞填充（自动版）

void borderClear(const cv::Mat &src, cv::Mat &dst);  // 边界清除
```

- **example.h**

```cpp
void holeFillExample();  // 孔洞填充示例

void borderClearExample();  // 边界清除示例

void morphFlattenBackgroundExample();  // 使用灰度级形态学重建展平复杂背景示例
```

### 10. 图像分割

- **image_segmentation.h**

```cpp
void pointDetectLaplaceKernel(const cv::Mat &src, cv::Mat &dst);  // 基于拉普拉斯核的孤立点检测

void lineDetectLaplaceKernel(const cv::Mat &src, cv::Mat &dst, int line_type);  // 基于拉普拉斯核的线检测

// 边缘检测：
//  1. 降低噪声
//  2. 检测边缘，可参考 “spatial_filter.h” 中的高通部分
//      基本方法：计算图像的导数，即空间高通滤波，例如 Sobel 算子等；
//      进阶方法：在滤波的基础上增加了对图像噪声和边缘性质等因素的考虑，例如 Canny 算子等

void lineDetectHough(const cv::Mat &src, cv::Mat &dst, double rho, double theta, int threshold, double srn = 0, double stn = 0, double min_theta = 0, double max_theta = CV_PI);  // 基于霍夫变换的线检测

void lineSegmentDetectHough(const cv::Mat &src, cv::Mat &dst, double rho, double theta, int threshold, double minLineLength = 0, double maxLineGap = 0);  // 基于霍夫变换的线段检测

void circleDetectHough(const cv::Mat &src, cv::Mat &dst, int method = cv::HOUGH_GRADIENT, double dp = 1, double minDist = 20, double param1 = 100, double param2 = 100, int minRadius = 0, int maxRadius = 0);  // 基于霍夫变换的圆检测

void cornerDetectHarris(const cv::Mat &src, cv::Mat &dst, int threshold, int blockSize, int ksize, double k = 0.04, int borderType = cv::BORDER_DEFAULT);  // 基于 Harris 算法的角点检测

void cornerDetectShiTomasi(const cv::Mat &src, cv::Mat &dst, int maxCorners, double qualityLevel, double minDistance, cv::InputArray mask = cv::noArray(), int blockSize = 3);  // 基于 Shi-Tomasi 算法的角点检测

void cornerDetectSubPixel(const cv::Mat &src, cv::Mat &dst, int maxCorners, double qualityLevel, double minDistance, cv::Size winSize, cv::Size zeroZone, cv::TermCriteria criteria, cv::InputArray mask = cv::noArray(), int blockSize = 3, bool useHarrisDetector = false, double k = 0.04);  // 亚像素级角点检测

int calcGlobalThresholdClassMean(const cv::Mat &src, const cv::Mat &mask = cv::Mat());  // 基于类间均值的全局（灰度分割）阈值处理

int calcGlobalThresholdOtus(const cv::Mat &src, const cv::Mat &mask = cv::Mat(), double *eta = nullptr);  // 基于大津法的全局（灰度分割）阈值处理

// OpenCV double otsu_thresh = cv::threshold(src, dst, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);  基于 Otsu 方法的最优全局阈值处理

int getPercentileGrayValue(const cv::Mat &src, double percentile = 0.997);  // 获取灰度图像的百分位灰度值

int calcGlobalThresholdEdgeOpt(const cv::Mat &src, int gradient_mode = 1, double percentile = 0.997, int threshold_mode = 1);  // 基于边缘改进全局阈值处理

pair<int, int> calcGlobalDualThresholdOtus(const cv::Mat &src, const cv::Mat &mask = cv::Mat(), double *eta = nullptr);  // 基于大津法的全局（灰度分割）双阈值处理

void thresholdThreeClass(const cv::Mat &src, cv::Mat &dst, int t1, int t2);  // 使用双阈值分割图像为三类
```

- **example.h**

```cpp
void globalThresholdEdgeOptExample();  // 基于边缘改进全局阈值处理示例
```

