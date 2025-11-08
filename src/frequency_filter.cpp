#include "frequency_filter.h"


void dftShift(cv::Mat &image) {
    if (image.empty()) {
        THROW_ARG_ERROR("Input image is empty.");
    }

    int cx = image.cols / 2;
    int cy = image.rows / 2;

    // 重新排列傅里叶图像的象限
    cv::Mat tmp;
    cv::Mat q0(image, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(image, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(image, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(image, cv::Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void spatialToFrequency(const cv::Mat &src, cv::Mat &dst_complex) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    // 对输入矩阵进行零填充到合适的尺寸，使用镜像填充
    int M = cv::getOptimalDFTSize(src.rows);
    int N = cv::getOptimalDFTSize(src.cols);
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, 0, M - src.rows, 0, N - src.cols, cv::BORDER_REFLECT_101,
                       cv::Scalar::all(0));

    // 将输入矩阵拓展为复数，实部虚部均为浮点型
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complex_image;
    cv::merge(planes, 2, complex_image);

    // 进行离散傅里叶变换
    cv::dft(complex_image, complex_image);

    // 重新排列傅里叶图像的象限，使原点位于图像中心
    dftShift(complex_image);

    complex_image.copyTo(dst_complex);
}

void splitFrequencyMagnitude(const cv::Mat &src_complex, cv::Mat &dst_magnitude) {
    if (src_complex.empty()) {
        THROW_ARG_ERROR("Input src complex image is empty!");
    }

    // 从复数输出矩阵中分离出实部（即幅值）
    cv::Mat planes[2];
    cv::Mat magnitude_image;
    cv::split(src_complex, planes);
    cv::magnitude(planes[0], planes[1], magnitude_image);

    // 对实部进行对数变换，即 compute log(1 + sqrt(Re(DFT(src))**2 + Im(DFT(src))**2))
    magnitude_image += cv::Scalar::all(1);
    cv::log(magnitude_image, magnitude_image);

    // 将频谱的行数和列数裁剪至偶数
    magnitude_image = magnitude_image(cv::Rect(0, 0, magnitude_image.cols & -2, magnitude_image.rows & -2));

    // 对幅值进行归一化操作，因为 type 为 float，所以用 cv::imshow() 时会将像素值乘以 255
    cv::normalize(magnitude_image, magnitude_image, 0, 1, cv::NORM_MINMAX);

    magnitude_image.copyTo(dst_magnitude);
}

void frequencyToSpatial(const cv::Mat &src_complex, cv::Mat &dst, const cv::Size &original_size) {
    if (src_complex.empty()) {
        THROW_ARG_ERROR("Input src complex image is empty!");
    }
    if (original_size.width <= 0 || original_size.height <= 0) {
        THROW_ARG_ERROR("Invalid `original_size`.");
    }

    cv::Mat dft_complex = src_complex.clone();

    // 重新排列傅里叶图像的象限，使原点位于图像四角
    dftShift(dft_complex);

    // 进行反离散傅里叶变换并直接输出实部
    cv::Mat idft_real;
    cv::idft(dft_complex, idft_real, cv::DFT_REAL_OUTPUT);

    // 裁剪至原尺寸
    idft_real = idft_real(cv::Rect(0, 0, original_size.width, original_size.height));

    idft_real.copyTo(dst);
}

cv::Mat idealLowPassFreqKernel(const cv::Size &size, int sigma) {
    int M = cv::getOptimalDFTSize(size.height);
    int N = cv::getOptimalDFTSize(size.width);
    cv::Size t_size = cv::Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 1, D(u, v) <= D0
                 0, D(u, v) >  D0
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = std::sqrt(std::pow((float) i - (float) t_size.height / 2, 2) +
                                 std::pow((float) j - (float) t_size.width / 2, 2));
            if (d <= sigma) {
                kernel.at<float>(i, j) = 1;
            } else {
                kernel.at<float>(i, j) = 0;
            }
        }
    }

    return kernel;
}

cv::Mat gaussLowPassFreqKernel(const cv::Size &size, int sigma) {
    int M = cv::getOptimalDFTSize(size.height);
    int N = cv::getOptimalDFTSize(size.width);
    cv::Size t_size = cv::Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = e^(-(D^2) / (2 * D0^2))
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = std::sqrt(std::pow((float) i - (float) t_size.height / 2, 2) +
                                 std::pow((float) j - (float) t_size.width / 2, 2));
            double h = std::exp(-1 * std::pow(d, 2) / (2 * std::pow(sigma, 2)));
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

cv::Mat bwLowPassFreqKernel(const cv::Size &size, int sigma, int order) {
    int M = cv::getOptimalDFTSize(size.height);
    int N = cv::getOptimalDFTSize(size.width);
    cv::Size t_size = cv::Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 1 / (1 + (D / D0)^(2n))
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = std::sqrt(std::pow((float) i - (float) t_size.height / 2, 2) +
                                 std::pow((float) j - (float) t_size.width / 2, 2));
            double h = 1 / (1 + std::pow(d / sigma, 2 * order));
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

cv::Mat idealHighPassFreqKernel(const cv::Size &size, int sigma) {
    int M = cv::getOptimalDFTSize(size.height);
    int N = cv::getOptimalDFTSize(size.width);
    cv::Size t_size = cv::Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 0, D(u, v) <= D0
                 1, D(u, v) >  D0
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = std::sqrt(std::pow((float) i - (float) t_size.height / 2, 2) +
                                 std::pow((float) j - (float) t_size.width / 2, 2));
            if (d <= sigma) {
                kernel.at<float>(i, j) = 0;
            } else {
                kernel.at<float>(i, j) = 1;
            }
        }
    }

    return kernel;
}

cv::Mat gaussHighPassFreqKernel(const cv::Size &size, int sigma) {
    int M = cv::getOptimalDFTSize(size.height);
    int N = cv::getOptimalDFTSize(size.width);
    cv::Size t_size = cv::Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 1 - e^(-(D^2) / (2 * D0^2))
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = std::sqrt(std::pow((float) i - (float) t_size.height / 2, 2) +
                                 std::pow((float) j - (float) t_size.width / 2, 2));
            double h = 1 - std::exp(-1 * std::pow(d, 2) / (2 * std::pow(sigma, 2)));
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

cv::Mat bwHighPassFreqKernel(const cv::Size &size, int sigma, int order) {
    int M = cv::getOptimalDFTSize(size.height);
    int N = cv::getOptimalDFTSize(size.width);
    cv::Size t_size = cv::Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 1 / (1 + (D0 / D)^(2n))
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = std::sqrt(std::pow((float) i - (float) t_size.height / 2, 2) +
                                 std::pow((float) j - (float) t_size.width / 2, 2));
            double h = 1 / (1 + std::pow(sigma / d, 2 * order));
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

cv::Mat highFreqEmphasisKernel(const cv::Size &size, int sigma, float k1, float k2) {
    int M = cv::getOptimalDFTSize(size.height);
    int N = cv::getOptimalDFTSize(size.width);
    cv::Size t_size = cv::Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = k1 + k2 * (1 - e^(-(D^2) / (2 * D0^2)))
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = std::sqrt(std::pow((float) i - (float) t_size.height / 2, 2) +
                                 std::pow((float) j - (float) t_size.width / 2, 2));
            double h = k1 + k2 * (1 - std::exp(-1 * std::pow(d, 2) / (2 * std::pow(sigma, 2))));
            kernel.at<float>(i, j) = (float) (h);
        }
    }

    return kernel;
}

cv::Mat homomorphicEmphasisKernel(const cv::Size &size, int sigma, float gamma_h, float gamma_l, int c) {
    if (gamma_h < 1) {
        THROW_ARG_ERROR("Invalid `gamma_h`: {}. You should make sure `gamma_h >= 1`.", gamma_h);
    }
    if (gamma_l <= 0 || gamma_l >= 1) {
        THROW_ARG_ERROR("Invalid `gamma_l`: {}. You should make sure `0 < gamma_l < 1`.", gamma_l);
    }

    int M = cv::getOptimalDFTSize(size.height);
    int N = cv::getOptimalDFTSize(size.width);
    cv::Size t_size = cv::Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = (gh - gl) * (1 - e^(-c * (D^2) / (D0^2))) + gl
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = std::sqrt(std::pow((float) i - (float) t_size.height / 2, 2) +
                                 std::pow((float) j - (float) t_size.width / 2, 2));
            double h = (gamma_h - gamma_l) * (1 - std::exp(-1 * c * std::pow(d, 2) / std::pow(sigma, 2))) + gamma_l;
            kernel.at<float>(i, j) = (float) (h);
        }
    }

    return kernel;
}

cv::Mat idealBandRejectFreqKernel(const cv::Size &size, int C0, int width) {
    int M = cv::getOptimalDFTSize(size.height);
    int N = cv::getOptimalDFTSize(size.width);
    cv::Size t_size = cv::Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 0, C0 - W/2 <= D(u, v) <= C0 + W/2
                 1, other
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = std::sqrt(std::pow((float) i - (float) t_size.height / 2, 2) +
                                 std::pow((float) j - (float) t_size.width / 2, 2));
            if ((C0 - width / 2.0 <= d) && (d <= C0 + width / 2.0)) {
                kernel.at<float>(i, j) = 0;
            } else {
                kernel.at<float>(i, j) = 1;
            }
        }
    }

    return kernel;
}

cv::Mat gaussBandRejectFreqKernel(const cv::Size &size, int C0, int width) {
    int M = cv::getOptimalDFTSize(size.height);
    int N = cv::getOptimalDFTSize(size.width);
    cv::Size t_size = cv::Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 1 - e^(-((D^2 - C0^2) / (D * W))^2)
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = std::sqrt(std::pow((float) i - (float) t_size.height / 2, 2) +
                                 std::pow((float) j - (float) t_size.width / 2, 2));
            double h = 1 - std::exp(-1 * std::pow((std::pow(d, 2) - std::pow(C0, 2)) / (d * width), 2));
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

cv::Mat bwBandRejectFreqKernel(const cv::Size &size, int C0, int width, int order) {
    int M = cv::getOptimalDFTSize(size.height);
    int N = cv::getOptimalDFTSize(size.width);
    cv::Size t_size = cv::Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 1 / (1 + ((D * W) / (D^2 - C0^2))^(2n))
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = std::sqrt(std::pow((float) i - (float) t_size.height / 2, 2) +
                                 std::pow((float) j - (float) t_size.width / 2, 2));
            double h = 1 / (1 + std::pow((d * width) / (std::pow(d, 2) - std::pow(C0, 2)), 2 * order));
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

void frequencyFilter(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel, bool rm_negative) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty!");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if (kernel.size() != src.size()) {
        THROW_ARG_ERROR("The input kernel size must be the same as the input src image size.");
    }

    // 转到频率域
    cv::Mat image_frequency;
    spatialToFrequency(src, image_frequency);

    // 分离
    cv::Mat planes[2];
    cv::split(image_frequency, planes);

    // 处理
    cv::Mat image_real, image_imaginary;
    cv::multiply(planes[0], kernel, image_real);
    cv::multiply(planes[1], kernel, image_imaginary);

    // 合并
    planes[0] = image_real;
    planes[1] = image_imaginary;
    cv::merge(planes, 2, image_frequency);

    // 转到空间域
    image_real = cv::Mat::zeros(src.size(), src.depth());
    frequencyToSpatial(image_frequency, image_real, src.size());

    // remove negative value 将负值置为零
    if (rm_negative) {
        for (int i = 0; i < image_real.rows; i++) {
            for (int j = 0; j < image_real.cols; j++) {
                float m = image_real.at<float>(i, j);
                if (m < 0) {
                    image_real.at<float>(i, j) = 0;
                }
            }
        }
    }

    // 对幅值进行归一化操作，转换为 CV_8U [0, 255]
    cv::normalize(image_real, image_real, 0, 255, cv::NORM_MINMAX, CV_8U);

    image_real.copyTo(dst);
}

cv::Mat laplaceFreqKernel(const cv::Size &size) {
    int M = cv::getOptimalDFTSize(size.height);
    int N = cv::getOptimalDFTSize(size.width);
    cv::Size t_size = cv::Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = -4 * pi^2 * D^2
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d2 = std::pow((float) i - (float) t_size.height / 2, 2) +
                        std::pow((float) j - (float) t_size.width / 2, 2);
            double h = -4 * std::pow(CV_PI, 2) * d2;
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

void freqSharpenLaplace(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty!");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    // 拉普拉斯锐化
    cv::Mat image_lf;
    cv::Mat kernel = laplaceFreqKernel(src.size());
    frequencyFilter(src, image_lf, kernel);

    // 增强：g(x, y) = f(x, y) - image_lf(x, y)
    cv::Mat image_g;
    cv::subtract(src, image_lf, image_g);

    image_g.copyTo(dst);
}

void frequencyFilterPlMul(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel, bool rm_negative) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty!");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if (kernel.size() != src.size()) {
        THROW_ARG_ERROR("The input kernel size must be the same as the input src image size.");
    }
    if (kernel.type() != CV_32FC2) {
        THROW_ARG_ERROR("Input kernel type must be CV_32FC2.");
    }

    // 转到频率域
    cv::Mat image_frequency;
    spatialToFrequency(src, image_frequency);

    // 频率域 复数乘法
    // a * b = (a1 + i*a2) * (b1 + i*b2)
    //       = a1*b1 + i*(a1*b2 + a2*b1) - b2*a2
    //       = (a1*b1 - a2*b2) + i*(a1*b2 + a2*b1)
    cv::mulSpectrums(image_frequency, kernel, image_frequency, 0);

    // 转到空间域
    cv::Mat image_real;
    image_real = cv::Mat::zeros(src.size(), src.depth());
    frequencyToSpatial(image_frequency, image_real, src.size());

    // remove negative value 将负值置为零
    if (rm_negative) {
        for (int i = 0; i < image_real.rows; i++) {
            for (int j = 0; j < image_real.cols; j++) {
                float m = image_real.at<float>(i, j);
                if (m < 0) {
                    image_real.at<float>(i, j) = 0;
                }
            }
        }
    }

    // 对幅值进行归一化操作，转换为 CV_8U [0, 255]
    cv::normalize(image_real, image_real, 0, 255, cv::NORM_MINMAX, CV_8U);

    image_real.copyTo(dst);
}

void bestNotchFilter(const cv::Mat &src, cv::Mat &dst, const cv::Mat &nbp_kernel, const cv::Size &opt_ksize) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty!");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if (nbp_kernel.size() != src.size()) {
        THROW_ARG_ERROR("The input nbp kernel size must be the same as the input src image size.");
    }
    if ((opt_ksize.width <= 0) || (opt_ksize.width % 2 == 0) || (opt_ksize.height <= 0) ||
        (opt_ksize.height % 2 == 0)) {
        THROW_ARG_ERROR("Invalid `opt_ksize`. You should make sure opt_ksize is positive odd number.");
    }

    // 通过 陷波带通滤波器 获取噪声图像
    cv::Mat noise;
    frequencyFilter(src, noise, nbp_kernel);  // TODO: rm_negative 参数 待测试

    int row_border = (opt_ksize.height - 1) / 2;
    int col_border = (opt_ksize.width - 1) / 2;

    cv::Mat src_copy;
    cv::copyMakeBorder(src, src_copy, row_border, row_border, col_border, col_border, cv::BORDER_REFLECT);
    cv::copyMakeBorder(noise, noise, row_border, row_border, col_border, col_border, cv::BORDER_REFLECT);

    cv::Mat temp = cv::Mat::zeros(src_copy.size(), src_copy.type());
    std::vector<int> g_values{};
    std::vector<int> n_values{};
    std::vector<int> gn_values{};
    std::vector<int> n2_values{};

    for (int i = row_border; i < src_copy.rows - row_border; i++) {
        for (int j = col_border; j < src_copy.cols - col_border; j++) {
            g_values.clear();
            n_values.clear();
            gn_values.clear();
            n2_values.clear();
            for (int k = -row_border; k <= row_border; k++) {
                for (int l = -col_border; l <= col_border; l++) {
                    int g_temp = src_copy.at<uchar>(i + k, j + l);
                    int n_temp = noise.at<uchar>(i + k, j + l);
                    g_values.emplace_back(g_temp);
                    n_values.emplace_back(n_temp);
                    gn_values.emplace_back(g_temp * n_temp);
                    n2_values.emplace_back(n_temp * n_temp);
                }
            }

            // 计算参数
            double g_mean =
                    (double) std::accumulate(std::begin(g_values), std::end(g_values), 0) / (double) g_values.size();
            double n_mean =
                    (double) std::accumulate(std::begin(n_values), std::end(n_values), 0) / (double) n_values.size();
            double gn_mean =
                    (double) std::accumulate(std::begin(gn_values), std::end(gn_values), 0) / (double) gn_values.size();
            double n2_mean =
                    (double) std::accumulate(std::begin(n2_values), std::end(n2_values), 0) / (double) n2_values.size();

            // 赋值
            temp.at<uchar>(i, j) = (int) (src_copy.at<uchar>(i, j) -
                                          ((gn_mean - g_mean * n_mean) / (n2_mean - n_mean * n_mean)) *
                                          noise.at<uchar>(i, j));
        }
    }

    cv::Mat temp_roi = temp(cv::Rect(col_border, row_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}
