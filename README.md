# DIP-OpenCV: 数字图像处理 OpenCV_Cpp

本项目基于冈萨雷斯的《数字图像处理》(第四版)，使用了 OpenCV_Cpp 对该书的一些理论知识进行了代码实现。



## 内容介绍（按章节）

### 3. 灰度变换与空间滤波

- 灰度变换 **gray_transform.h**

```cpp
void bgrToGray(Mat &src, Mat &dst);  // BGR 转换为 GRAY

void grayLinearScaleCV_8U(Mat &src, Mat &dst);  // 灰度线性缩放，缩放至 [0-255]

void grayInvert(Mat &src, Mat &dst);  // 灰度反转（属于灰度线性变换）

void grayLog(Mat &src, Mat &dst);  // 灰度对数变换

void grayAntiLog(Mat &src, Mat &dst);  // 灰度反对数变换

void grayGamma(Mat &src, Mat &dst, float gamma);  // 灰度伽马变换，也称幂律变换

void grayContrastStretch(Mat &src, Mat &dst, uint r1, uint s1, uint r2, uint s2);  // 灰度对比度拉伸

void grayLayering(Mat &src, Mat &dst, uint r1, uint r2, uint s, bool other_zero);  // 灰度值级分层

void grayBitPlaneLayering(Mat &src, vector<Mat> &dst);  // 灰度比特平面分层

Mat grayHistogram(Mat &src, Size size = Size(512, 400), const Scalar &color = Scalar(255, 255, 255));  // 灰度直方图，即单通道直方图

// OpenCV void equalizeHist( InputArray src, OutputArray dst );  全局直方图均衡化

void localEqualizeHist(Mat &src, Mat &dst, double clipLimit = 40.0, Size tileGridSize = Size(8, 8));  // 局部直方图均衡化

void matchHist(Mat &src, Mat &dst, Mat &refer);  // 直方图规定化

void shadingCorrection(Mat &src, Mat &dst, float k1 = 0.25, float k2 = 6);  // 阴影校正
```

- 空间滤波 **spatial_filter.h**

```cpp
void linearSpatialFilter(Mat &src, Mat &dst, Mat &kernel);  // 线性空间滤波（即二维图像卷积）

void smoothSpatialFilterBox(Mat &src, Mat &dst, Size ksize, Point anchor = Point(-1, -1), bool normalize = true, int borderType = BORDER_DEFAULT);  // 盒式平滑（低通）空间滤波

void smoothSpatialFilterGauss(Mat &src, Mat &dst, Size ksize, double sigmaX, double sigmaY = 0, int borderType = BORDER_DEFAULT);  // 高斯平滑（低通）空间滤波

void orderStatisticsFilter(Mat &src, Mat &dst, int ksize, int percentage = 50);  // 统计排序（非线性）滤波器

void sharpenSpatialFilterLaplace(Mat &src, Mat &dst, int ksize = 1, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT);  // 拉普拉斯（二阶导数）锐化（高通）空间滤波

void sharpenSpatialFilterTemplate(Mat &src, Mat &dst, Size smooth_size, float k = 1);  // 模板锐化（钝化掩蔽、高提升滤波）

void sharpenSpatialFilterRoberts();  // Roberts 算子 TODO:

void sharpenSpatialFilterPrewitt();  // Prewitt 算子 TODO:

void sharpenSpatialFilterSobel(Mat &src, Mat &dst, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT);  // 索贝尔（一阶导数）锐化（高通）空间滤波

void sharpenSpatialFilterScharr(Mat &src, Mat &dst, int dx, int dy, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT);  // 沙尔（一阶导数）锐化（高通）空间滤波

void sharpenSpatialFilterCanny(Mat &src, Mat &dst, double threshold1, double threshold2, int apertureSize = 3, bool L2gradient = false);  // Canny 锐化（高通）空间滤波（边缘检测优化算法）
```

- **example.h**

```cpp
void localEqualizeHistExample();  // 局部直方图均衡化示例

void shadingCorrectionExample();  // 阴影校正示例
```

### 4. 频率域滤波

- **frequency_filter.h**

```cpp
void dftShift(Mat &image);  // 傅里叶图像象限变换

void spatialToFrequency(Mat &src, Mat &dst_complex);  // 图像空间域转频率域

void splitFrequencyMagnitude(Mat &src_complex, Mat &dst_magnitude);  // 从频率域复数图像中分离出频率域实部幅值图像

void frequencyToSpatial(Mat &src_complex, Mat &dst);  // 图像频率域转空间域

Mat idealLowPassFreqKernel(Size size, int sigma);  // 理想低通频率滤波核函数，该核有振铃效应

Mat gaussLowPassFreqKernel(Size size, int sigma);  // 高斯低通频率滤波核函数

Mat bwLowPassFreqKernel(Size size, int sigma, int order);  // 巴特沃斯低通频率滤波核函数

Mat idealHighPassFreqKernel(Size size, int sigma);  // 理想高通频率滤波核函数，该核有振铃效应

Mat gaussHighPassFreqKernel(Size size, int sigma);  // 高斯高通频率滤波核函数

Mat bwHighPassFreqKernel(Size size, int sigma, int order);  // 巴特沃斯高通频率滤波核函数

Mat highFreqEmphasisKernel(Size size, int sigma, float k1 = 1, float k2 = 1);  // 高频增强滤波核函数

Mat homomorphicEmphasisKernel(Size size, int sigma, float gamma_h, float gamma_l, int c);  // 同态增强滤波核函数

Mat idealBandRejectFreqKernel(Size size, int C0, int width);  // 理想带阻频率滤波核函数

Mat gaussBandRejectFreqKernel(Size size, int C0, int width);  // 高斯带阻频率滤波核函数

Mat bwBandRejectFreqKernel(Size size, int C0, int width, int order);  // 巴特沃斯带阻频率滤波核函数

// Mat notchBandRejectFreqKernel();  陷波带阻滤波核（定制化，主要用于处理周期噪声）

void frequencyFilter(Mat &src, Mat &dst, Mat &kernel, bool rm_negative = false);  // 频率域滤波

Mat laplaceFreqKernel(Size size);  // 拉普拉斯频率滤波核函数

void freqSharpenLaplace(Mat &src, Mat &dst);  // 拉普拉斯频率域锐化

void frequencyFilterPlMul(Mat &src, Mat &dst, Mat &kernel, bool rm_negative = false);  // 频率域滤波（复数乘法版）
```

- **example.h**

```cpp
void domainTransformExample();  // 空间域图像与频率域图像的转换示例

void highFreqEmphasisExample();  // 高频增强滤波示例
```

### 5. 图像复原与重构

- **image_noise.h**

```cpp
void addNoiseGauss(Mat &src, Mat &dst, int mean, int sigma);  // 添加高斯噪声

void addNoiseMean(Mat &src, Mat &dst, int lower, int upper);  // 添加平均噪声

void addNoiseRayleigh(Mat &src, Mat &dst, double sigma);  // 添加瑞利噪声

void addNoiseGamma(Mat &src, Mat &dst, double sigma, double alpha, double beta);  // 添加伽马(爱尔兰）噪声

void addNoiseExp(Mat &src, Mat &dst, double lambda);  // 添加指数噪声

void addNoiseSaltPepper(Mat &src, Mat &dst, double noise_level, double salt_value = 255, double pepper_value = 0);  // 添加椒盐(冲激)噪声
```

- **spatial_filter.h**

```cpp
void geometricMeanFilter(Mat &src, Mat &dst, Size ksize);  // 几何均值滤波器，效果优于算术平均滤波器（即盒式滤波器）

void harmonicAvgFilter(Mat &src, Mat &dst, Size ksize);  // 谐波平均滤波器，能够处理 盐粒噪声 或 类高斯噪声，不能处理 胡椒噪声

void antiHarmonicAvgFilter(Mat &src, Mat &dst, Size ksize, float order);  // 反谐波平均滤波器，能够处理 盐粒噪声 或 胡椒噪声 或 类高斯噪声

void midPointFilter(Mat &src, Mat &dst, Size ksize);  // 中点滤波器，适合处理随机分布的噪声，如 高斯噪声 或 均匀噪声

void modifiedAlphaMeanFilter(Mat &src, Mat &dst, Size ksize, int d);  // 修正阿尔法滤波器，适合处理多种混合噪声

void adaptiveLocalFilter(Mat &src, Mat &dst, Size ksize);  // 自适应局部降噪滤波器

void adaptiveMedianFilter(Mat &src, Mat &dst, int max_ksize);  // 自适应中值滤波器，能够去除椒盐噪声、平滑其他非冲激噪声且减少失真
```

- **frequency_filter.h**

```cpp
void bestNotchFilter(Mat &src, Mat &dst, Mat &nbp_kernel, Size opt_ksize);  // 最优陷波滤波
```

### 6. 彩色图像处理

- **color_process.h**

```cpp
vector<Mat> colorChannelSpilt(Mat &src);  // 彩色通道分离

void bgrToHsi(Mat &src, Mat &dst);  // BGR 转换为 HSI

void hsiToBgr(Mat &src, Mat &dst);  // HSI 转换为 BGR

void pseudoColor(Mat &src, Mat &dst, ColormapTypes color = COLORMAP_JET);  // 伪彩色处理

void complementaryColor(Mat &src, Mat &dst);  // 补色处理，即彩色反转

void colorLayering(Mat &src, Mat &dst, const Vec3b& color_center, double range_radius = 120);  // 彩色分层

// 彩色图像（RGB / HSI）的校正（对数变换 / 反对数变换 / 伽马变换）：参考灰度

void colorEqualizeHist(Mat &src, Mat &dst);  // 彩色全局直方图均衡化（不建议使用）
```

- **example.h**

```cpp
void hsiExample();  // HSI 转换示例
```

### 7. 小波变换和其他图像变换

- **wavelet_transform.h**

```cpp
void DWT(Mat &src, Mat &dst, const string &wname, int level);  // 离散小波变换

void IDWT(Mat &src, Mat &dst, const string &wname, int level);  // 离散小波逆变换

void DCT(Mat &src, Mat &dst);  // 离散余弦变换

void IDCT(Mat &src, Mat &dst);  // 离散余弦逆变换

void blockDCT(Mat &src, Mat &dst, int block_size = 8);  // 分块离散余弦变换

void blockIDCT(Mat &src, Mat &dst, int block_size = 8);  // 分块离散余弦逆变换
```

- **example.h**

```cpp
void DCTExample();  // 离散余弦变换及逆变换示例

void blockDCTExample();  // 分块离散余弦变换及逆变换示例
```

### 9. 形态学图像处理

- **morphological.h**

```cpp
void grayToBinary(Mat &src, Mat &dst, double thresh, double maxval, int type);  // GRAY 转换为 Binary (二值化)

uchar getBinaryMaxval(Mat &src);  // 获取二值图像的最大值

void binaryInvert(Mat &src, Mat &dst);  // 二值反转

// OpenCV Mat getStructuringElement(int shape, Size ksize, Point anchor = Point(-1,-1));  构建（形态学）结构元

void morphologyErode(Mat &src, Mat &dst, const Mat &kernel);  // 形态学腐蚀

void morphologyDilate(Mat &src, Mat &dst, const Mat &kernel);  // 形态学膨胀

void morphologyOpen(Mat &src, Mat &dst, const Mat &kernel);  // 形态学开运算

void morphologyClose(Mat &src, Mat &dst, const Mat &kernel);  // 形态学闭运算

void morphologyHMT(Mat &src, Mat &dst, const Mat &fore_kernel, const Mat &back_kernel);  // 形态学击中击不中变换

void morphologyGradient(Mat &src, Mat &dst, const Mat &kernel);  // 形态学梯度

void morphologyTophat(Mat &src, Mat &dst, const Mat &kernel);  // 形态学顶帽变换

void morphologyBlackhat(Mat &src, Mat &dst, const Mat &kernel);  // 形态学底帽变换

void boundaryExtract(Mat &src, Mat &dst, int size);  // 边界提取

void holeFill(Mat &src, Mat &dst, Mat &start);  // 孔洞填充

void extractConnected(Mat &src, Mat &dst);  // 提取连通分量

// 凸壳、细化、粗化、骨架、裁剪

void erodeReconstruct(Mat &src, const Mat &tmpl, Mat &dst);  // 腐蚀形态学重建

void dilateReconstruct(Mat &src, const Mat &tmpl, Mat &dst);  // 膨胀形态学重建

void openReconstruct(Mat &src, Mat &dst, const Mat &erode_kernel, int erode_times = 1);  // 开运算形态学重建

void closeReconstruct(Mat &src, Mat &dst, const Mat &dilate_kernel, int dilate_times = 1);  // 闭运算形态学重建

void tophatReconstruct(Mat &src, Mat &dst, const Mat &erode_kernel, int erode_times = 1);  // 顶帽形态学重建

void blackhatReconstruct(Mat &src, Mat &dst, const Mat &dilate_kernel, int dilate_times = 1);  // 底帽形态学重建

void holeFill(Mat &src, Mat &dst);  // 孔洞填充（自动版）

void borderClear(Mat &src, Mat &dst);  // 边界清除
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
void pointDetectLaplaceKernel(Mat &src, Mat &dst);  // 基于拉普拉斯核的孤立点检测

void lineDetectLaplaceKernel(Mat &src, Mat &dst, int line_type);  // 基于拉普拉斯核的线检测

// 边缘检测：
//  1. 降低噪声
//  2. 检测边缘，可参考 “spatial_filter.h” 中的高通部分
//      基本方法：计算图像的导数，即空间高通滤波，例如 Sobel 算子等；
//      进阶方法：在滤波的基础上增加了对图像噪声和边缘性质等因素的考虑，例如 Canny 算子等

void lineDetectHough(Mat &src, Mat &dst, double rho, double theta, int threshold, double srn = 0, double stn = 0, double min_theta = 0, double max_theta = CV_PI);  // 基于霍夫变换的线检测

void lineSegmentDetectHough(Mat &src, Mat &dst, double rho, double theta, int threshold, double minLineLength = 0, double maxLineGap = 0);  // 基于霍夫变换的线段检测

void cornerDetectHarris(Mat &src, Mat &dst, int threshold, int blockSize, int ksize, double k = 0.04, int borderType = BORDER_DEFAULT);  // 基于 Harris 算法的角点检测

void cornerDetectShiTomasi(Mat &src, Mat &dst, int maxCorners, double qualityLevel, double minDistance, InputArray mask = noArray(), int blockSize = 3);  // 基于 Shi-Tomasi 算法的角点检测

void cornerDetectSubPixel(Mat &src, Mat &dst, int maxCorners, double qualityLevel, double minDistance, Size winSize, Size zeroZone, TermCriteria criteria, InputArray mask = noArray(), int blockSize = 3, bool useHarrisDetector = false, double k = 0.04);  // 亚像素级角点检测

void keyPointDetectSurf(Mat &src, Mat &dst, double hessianThreshold = 100, int nOctaves = 4, int nOctaveLayers = 3, bool extended = false, bool upright = false);  // 基于 SURF 算法的关键点特征检测
```

