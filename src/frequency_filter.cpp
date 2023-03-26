#include "frequency_filter.h"


void dftShift(Mat &image) {
    int cx = image.cols / 2;
    int cy = image.rows / 2;

    // 重新排列傅里叶图像的象限
    Mat tmp;
    Mat q0(image, Rect(0, 0, cx, cy));
    Mat q1(image, Rect(cx, 0, cx, cy));
    Mat q2(image, Rect(0, cy, cx, cy));
    Mat q3(image, Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void spatialToFrequency(Mat &src, Mat &dst_complex) {
    if (src.empty()) {
        throw invalid_argument("spatialToFrequency(): Input image is empty!");
    }

    // 对输入矩阵进行零填充到合适的尺寸，使用镜像填充
    int M = getOptimalDFTSize(src.rows);
    int N = getOptimalDFTSize(src.cols);
    Mat padded;
    copyMakeBorder(src, padded, 0, M - src.rows, 0, N - src.cols, BORDER_REFLECT_101,
                   Scalar::all(0));

    // 将输入矩阵拓展为复数，实部虚部均为浮点型
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complex_image;
    merge(planes, 2, complex_image);

    // 进行离散傅里叶变换
    dft(complex_image, complex_image);

    // 重新排列傅里叶图像的象限，使原点位于图像中心
    dftShift(complex_image);

    complex_image.copyTo(dst_complex);
}

void splitFrequencyMagnitude(Mat &src_complex, Mat &dst_magnitude) {
    if (src_complex.empty()) {
        throw invalid_argument("splitFrequencyMagnitude(): Input image is empty!");
    }

    // 从复数输出矩阵中分离出实部（即幅值）
    Mat planes[2];
    Mat magnitude_image;
    split(src_complex, planes);
    magnitude(planes[0], planes[1], magnitude_image);

    // 对实部进行对数变换，即 compute log(1 + sqrt(Re(DFT(src))**2 + Im(DFT(src))**2))
    magnitude_image += Scalar::all(1);
    log(magnitude_image, magnitude_image);

    // 将频谱的行数和列数裁剪至偶数
    magnitude_image = magnitude_image(Rect(0, 0, magnitude_image.cols & -2, magnitude_image.rows & -2));

    // 对幅值进行归一化操作，因为 type 为 float，所以用 imshow() 时会将像素值乘以 255
    normalize(magnitude_image, magnitude_image, 0, 1, NORM_MINMAX);

    magnitude_image.copyTo(dst_magnitude);
}

void frequencyToSpatial(Mat &src_complex, Mat &dst) {
    if (src_complex.empty()) {
        throw invalid_argument("frequencyToSpatial(): Input image is empty!");
    }

    // 为确保输出图像与原图像的尺寸一致，应初始化并传入正确尺寸的 dst 对象，
    // 如：Mat dst = Mat::zeros(src.size(), src.depth());
    if (dst.empty()) {
        throw invalid_argument("frequencyToSpatial(): Dst size unknown!");
    }

    Mat dft_complex = src_complex.clone();

    // 重新排列傅里叶图像的象限，使原点位于图像四角
    dftShift(dft_complex);

    // 进行反离散傅里叶变换并直接输出实部
    Mat idft_real;
    idft(dft_complex, idft_real, DFT_REAL_OUTPUT);

    idft_real.copyTo(dst);
}

void domainTransformDemo() {
    Mat image_gray = imread(R"(..\image\barbara.tif)", 0);

    Mat dst_complex, dst_magnitude;
    spatialToFrequency(image_gray, dst_complex);
    splitFrequencyMagnitude(dst_complex, dst_magnitude);

    Mat dst_idft = Mat::zeros(image_gray.size(), image_gray.depth());
    frequencyToSpatial(dst_complex, dst_idft);

    namedWindow("src", WINDOW_AUTOSIZE);
    imshow("src", image_gray);
    namedWindow("dft", WINDOW_AUTOSIZE);
    imshow("dft", dst_magnitude);
    namedWindow("idft", WINDOW_AUTOSIZE);
    imshow("idft", dst_idft);
    waitKey(0);
}

Mat idealLowPassFreqKernel(Size size, int sigma) {
    int M = getOptimalDFTSize(size.height);
    int N = getOptimalDFTSize(size.width);
    Size t_size = Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 1, D(u, v) <= D0
                 0, D(u, v) >  D0
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = sqrt(
                    pow((float) i - (float) t_size.height / 2, 2) + pow((float) j - (float) t_size.width / 2, 2));
            if (d <= sigma) {
                kernel.at<float>(i, j) = 1;
            } else {
                kernel.at<float>(i, j) = 0;
            }
        }
    }

    return kernel;
}

Mat gaussLowPassFreqKernel(Size size, int sigma) {
    int M = getOptimalDFTSize(size.height);
    int N = getOptimalDFTSize(size.width);
    Size t_size = Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = e^(-(D^2) / (2 * D0^2))
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = sqrt(
                    pow((float) i - (float) t_size.height / 2, 2) + pow((float) j - (float) t_size.width / 2, 2));
            double h = exp(-1 * pow(d, 2) / (2 * pow(sigma, 2)));
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

Mat bwLowPassFreqKernel(Size size, int sigma, int order) {
    int M = getOptimalDFTSize(size.height);
    int N = getOptimalDFTSize(size.width);
    Size t_size = Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 1 / (1 + (D / D0)^(2n))
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = sqrt(
                    pow((float) i - (float) t_size.height / 2, 2) + pow((float) j - (float) t_size.width / 2, 2));
            double h = 1 / (1 + pow(d / sigma, 2 * order));
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

Mat idealHighPassFreqKernel(Size size, int sigma) {
    int M = getOptimalDFTSize(size.height);
    int N = getOptimalDFTSize(size.width);
    Size t_size = Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 0, D(u, v) <= D0
                 1, D(u, v) >  D0
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = sqrt(
                    pow((float) i - (float) t_size.height / 2, 2) + pow((float) j - (float) t_size.width / 2, 2));
            if (d <= sigma) {
                kernel.at<float>(i, j) = 0;
            } else {
                kernel.at<float>(i, j) = 1;
            }
        }
    }

    return kernel;
}

Mat gaussHighPassFreqKernel(Size size, int sigma) {
    int M = getOptimalDFTSize(size.height);
    int N = getOptimalDFTSize(size.width);
    Size t_size = Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 1 - e^(-(D^2) / (2 * D0^2))
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = sqrt(
                    pow((float) i - (float) t_size.height / 2, 2) + pow((float) j - (float) t_size.width / 2, 2));
            double h = 1 - exp(-1 * pow(d, 2) / (2 * pow(sigma, 2)));
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

Mat bwHighPassFreqKernel(Size size, int sigma, int order) {
    int M = getOptimalDFTSize(size.height);
    int N = getOptimalDFTSize(size.width);
    Size t_size = Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 1 / (1 + (D0 / D)^(2n))
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = sqrt(
                    pow((float) i - (float) t_size.height / 2, 2) + pow((float) j - (float) t_size.width / 2, 2));
            double h = 1 / (1 + pow(sigma / d, 2 * order));
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

Mat highFreqEmphasisKernel(Size size, int sigma, float k1, float k2) {
    int M = getOptimalDFTSize(size.height);
    int N = getOptimalDFTSize(size.width);
    Size t_size = Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = k1 + k2 * (1 - e^(-(D^2) / (2 * D0^2)))
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = sqrt(
                    pow((float) i - (float) t_size.height / 2, 2) + pow((float) j - (float) t_size.width / 2, 2));
            double h = k1 + k2 * (1 - exp(-1 * pow(d, 2) / (2 * pow(sigma, 2))));
            kernel.at<float>(i, j) = (float) (h);
        }
    }

    return kernel;
}

Mat homomorphicEmphasisKernel(Size size, int sigma, float gamma_h, float gamma_l, int c) {
    int M = getOptimalDFTSize(size.height);
    int N = getOptimalDFTSize(size.width);
    Size t_size = Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = (gh - gl) * (1 - e^(-c * (D^2) / (D0^2))) + gl
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = sqrt(
                    pow((float) i - (float) t_size.height / 2, 2) + pow((float) j - (float) t_size.width / 2, 2));
            double h = (gamma_h - gamma_l) * (1 - exp(-1 * c * pow(d, 2) / pow(sigma, 2))) + gamma_l;
            kernel.at<float>(i, j) = (float) (h);
        }
    }

    return kernel;
}

Mat idealBandRejectFreqKernel(Size size, int C0, int width) {
    int M = getOptimalDFTSize(size.height);
    int N = getOptimalDFTSize(size.width);
    Size t_size = Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 0, C0 - W/2 <= D(u, v) <= C0 + W/2
                 1, other
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = sqrt(
                    pow((float) i - (float) t_size.height / 2, 2) + pow((float) j - (float) t_size.width / 2, 2));
            if ((C0 - width / 2.0 <= d) && (d <= C0 + width / 2.0)) {
                kernel.at<float>(i, j) = 0;
            } else {
                kernel.at<float>(i, j) = 1;
            }
        }
    }

    return kernel;
}

Mat gaussBandRejectFreqKernel(Size size, int C0, int width) {
    int M = getOptimalDFTSize(size.height);
    int N = getOptimalDFTSize(size.width);
    Size t_size = Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 1 - e^(-((D^2 - C0^2) / (D * W))^2)
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = sqrt(
                    pow((float) i - (float) t_size.height / 2, 2) + pow((float) j - (float) t_size.width / 2, 2));
            double h = 1 - exp(-1 * pow((pow(d, 2) - pow(C0, 2)) / (d * width), 2));
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

Mat bwBandRejectFreqKernel(Size size, int C0, int width, int order) {
    int M = getOptimalDFTSize(size.height);
    int N = getOptimalDFTSize(size.width);
    Size t_size = Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = 1 / (1 + ((D * W) / (D^2 - C0^2))^(2n))
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d = sqrt(
                    pow((float) i - (float) t_size.height / 2, 2) + pow((float) j - (float) t_size.width / 2, 2));
            double h = 1 / (1 + pow((d * width) / (pow(d, 2) - pow(C0, 2)), 2 * order));
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

void frequencyFilter(Mat &src, Mat &dst, Mat &kernel, bool rm_negative) {
    if (src.empty()) {
        throw invalid_argument("frequencyFilter(): Input image is empty!");
    }

    // 转到频率域
    Mat image_frequency;
    spatialToFrequency(src, image_frequency);

    // 分离
    Mat planes[2];
    split(image_frequency, planes);

    // 处理
    Mat image_real, image_imaginary;
    multiply(planes[0], kernel, image_real);
    multiply(planes[1], kernel, image_imaginary);

    // 合并
    planes[0] = image_real;
    planes[1] = image_imaginary;
    merge(planes, 2, image_frequency);

    // 转到空间域
    image_real = Mat::zeros(src.size(), src.depth());
    frequencyToSpatial(image_frequency, image_real);

    // remove negative value 将负值置为零
    if (rm_negative) {
        for (int i = 0; i < image_real.rows; i++) {
            for (int j = 0; j < image_real.cols; j++) {
                float m = image_real.at<float>(i, j);
                if (m < 0) image_real.at<float>(i, j) = 0;
            }
        }
    }

    // 对幅值进行归一化操作，转换为 CV_8U 0-255
    normalize(image_real, image_real, 0, 255, NORM_MINMAX, CV_8U);

    // 裁剪至原尺寸
    image_real = image_real(Rect(0, 0, src.cols, src.rows));

    image_real.copyTo(dst);
}

Mat laplaceFreqKernel(Size size) {
    int M = getOptimalDFTSize(size.height);
    int N = getOptimalDFTSize(size.width);
    Size t_size = Size(N, M);

    cv::Mat kernel(t_size, CV_32FC1);

    /* 传递函数：
       H(u, v) = -4 * pi^2 * D^2
       D(u, v) = [(u - P/2)^2 + (v - Q/2)^2]^(1/2)
    */
    for (int i = 0; i < t_size.height; i++) {
        for (int j = 0; j < t_size.width; j++) {
            double d2 = pow((float) i - (float) t_size.height / 2, 2) + pow((float) j - (float) t_size.width / 2, 2);
            double h = -4 * pow(M_PI, 2) * d2;
            kernel.at<float>(i, j) = (float) h;
        }
    }

    return kernel;
}

void freqSharpenLaplace(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("freqSharpenLaplace(): Input image is empty!");
    }

    // 拉普拉斯锐化
    Mat temp;
    Mat kernel = laplaceFreqKernel(src.size());
    frequencyFilter(src, temp, kernel);

    // 增强：g(x, y) = f(x, y) - lf(x, y)
    subtract(src, temp, dst);
}
