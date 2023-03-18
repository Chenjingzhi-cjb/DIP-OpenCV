# DIP-OpenCV：数字图像处理 OpenCV_Cpp

本项目基于冈萨雷斯的《数字图像处理》(第四版)，使用了 OpenCV_Cpp 对该书的一些理论知识进行了代码实现。



## 内容介绍（按章节）

### 3. 灰度变换与空间滤波

- 灰度变换 **gray_transform.h**

```cpp
void bgrToGray(Mat &src, Mat &dst);  // BGR 转换为 GRAY

void grayLinearScaleCV_8U(Mat &src, Mat &dst);  // 灰度线性缩放，缩放至 [0-255]

void grayInvert(Mat &src, Mat &dst);  // 灰度反转（属于灰度线性变换）

void grayLog(Mat &src, Mat &dst, float k);  // 灰度对数变换

void grayAntiLog(Mat &src, Mat &dst, float k);  // 灰度反对数变换

void grayGamma(Mat &src, Mat &dst, float k, float gamma);  // 灰度伽马变换，也称幂律变换

void grayContrastStretch(Mat &src, Mat &dst, uint r1, uint s1, uint r2, uint s2);  // 灰度对比度拉伸

void grayLayering(Mat &src, Mat &dst, uint r1, uint r2, uint s, bool other_zero);  // 灰度值级分层

void grayBitPlaneLayering(Mat &src, vector<Mat> &dst);  // 灰度比特平面分层
```

- 空间滤波 **spatial_filter.h**

```cpp
void linearSpatialFilter(Mat &src, Mat &dst, Mat &kernel);  // 线性空间滤波（即二维图像卷积）

void smoothSpatialFilterBox(Mat &src, Mat &dst, Size ksize, Point anchor = Point(-1, -1), bool normalize = true, int borderType = BORDER_DEFAULT);  // 盒式平滑（低通）空间滤波

void smoothSpatialFilterGauss(Mat &src, Mat &dst, Size ksize, double sigmaX, double sigmaY = 0, int borderType = BORDER_DEFAULT);  // 高斯平滑（低通）空间滤波

void shadingCorrection(Mat &src, Mat &dst, float k1, float k2);  // 阴影校正 TODO:

void orderStatisticsFilter(Mat &src, Mat &dst, int ksize, int percentage = 50);  // 统计排序（非线性）滤波器

void sharpenSpatialFilterLaplace(Mat &src, Mat &dst, int ksize = 1, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT);  // 拉普拉斯（二阶导数）锐化（高通）空间滤波

void sharpenSpatialFilterTemplate(Mat &src, Mat &dst, Size smooth_size, float k = 1);  // 模板锐化（钝化掩蔽、高提升滤波）

void sharpenSpatialFilterRoberts();  // Roberts 算子 TODO:

void sharpenSpatialFilterPrewitt();  // Prewitt 算子 TODO:

void sharpenSpatialFilterSobel(Mat &src, Mat &dst, int dx, int dy, int ksize = 3, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT);  // 索贝尔（一阶导数）锐化（高通）空间滤波

void sharpenSpatialFilterScharr(Mat &src, Mat &dst, int dx, int dy, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT);  // 沙尔（一阶导数）锐化（高通）空间滤波

void sharpenSpatialFilterCanny(Mat &src, Mat &dst, double threshold1, double threshold2, int apertureSize = 3, bool L2gradient = false);  // Canny 锐化（高通）空间滤波
```

### 4. 频率域滤波

- **spatial_filter.h**

```cpp
void dftShift(Mat &image);  // 傅里叶图像象限变换

void spatialToFrequency(Mat &src, Mat &dst_complex);  // 图像空间域转频率域

void splitFrequencyMagnitude(Mat &src_complex, Mat &dst_magnitude);  // 从频率域复数图像中分离出频率域实部幅值图像

void frequencyToSpatial(Mat &src_complex, Mat &dst);  // 图像频率域转空间域

void domainTransformDemo();  // 空间域图像与频率域图像的转换演示

Mat idealLowFrequencyKernel(Size size, float sigma);  // 理想低通频率滤波核函数

Mat gaussLowFrequencyKernel(Size size, float sigma);  // 高斯低通频率滤波核函数

Mat bwLowFrequencyKernel(Size size, float sigma, int order);  // 巴特沃斯低通频率滤波核函数

void smoothFrequencyFilter(Mat &src, Mat &dst, Mat &kernel);  // 平滑（低通）频率滤波
```

