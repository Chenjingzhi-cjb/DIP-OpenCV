#include "gray_transform.h"


void bgrToGray(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC3) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC3.");
    }

    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
}

void grayLinearScaleCV_8U(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::normalize(src, dst, 0, 255, cv::NORM_MINMAX, CV_8U);
}

void grayInvert(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    src.convertTo(dst, CV_8U, -1, 255);
}

void grayLog(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::Mat temp(src.size(), CV_32FC1);
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            int m = src.at<uchar>(r, c);
            // 进行 对数变换
            temp.at<float>(r, c) = (float) std::log10(1 + m);
        }
    }

    // 线性缩放至 [0, 255]
    cv::normalize(temp, temp, 0, 255, cv::NORM_MINMAX, CV_8U);

    temp.copyTo(dst);
}

void grayAntiLog(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::Mat temp(src.size(), CV_32FC1);
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            int m = src.at<uchar>(r, c);
            // 进行 反对数变换
            temp.at<float>(r, c) = (float) std::pow(10, (float) m / 255) - 1;
        }
    }

    // 线性缩放至 [0, 255]
    cv::normalize(temp, temp, 0, 255, cv::NORM_MINMAX, CV_8U);

    temp.copyTo(dst);
}

void grayGamma(const cv::Mat &src, cv::Mat &dst, float gamma) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if (gamma <= 0) {
        THROW_ARG_ERROR("Invalid `gamma`: {}. You should make sure `gamma > 0`.", gamma);
    }

    cv::Mat temp(src.size(), CV_32FC1);
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            int m = src.at<uchar>(r, c);
            // 进行 伽马变换
            temp.at<float>(r, c) = (float) std::pow(m, gamma);
        }
    }

    // 线性缩放至 [0, 255]
    cv::normalize(temp, temp, 0, 255, cv::NORM_MINMAX, CV_8U);

    temp.copyTo(dst);
}

void grayContrastStretch(const cv::Mat &src, cv::Mat &dst, uint r1, uint s1, uint r2, uint s2) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    // CV_8U -> [0, 255]
    const int upper_limit = 255;

    if (r1 >= r2 || s1 >= s2) {
        THROW_ARG_ERROR("Invalid `r1, r2, s1, s2`. You should make sure `r1 < r2` and `s1 < s2`.");
    }
    if (r1 <= 0 || s1 <= 0) {
        THROW_ARG_ERROR("Invalid `r1, s1`. You should make sure `r1, s1 > 0`.");
    }
    if (r2 >= upper_limit || s2 >= upper_limit) {
        THROW_ARG_ERROR("Invalid `r2, s2`. You should make sure `r2, s2 < {}`.", upper_limit);
    }

    double k1 = (double) s1 / r1;
    double k2 = (double) (s2 - s1) / (r2 - r1);
    double b2 = s1 - k2 * r1;
    double k3 = (double) (upper_limit - s2) / (upper_limit - r2);
    double b3 = s2 - k3 * r2;

    cv::Mat temp(src.size(), CV_8UC1);
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            int m = src.at<uchar>(r, c);
            // 进行 对比度拉伸
            double result;
            if (m < r1) {
                result = k1 * m;
            } else if (m < r2) {
                result = k2 * m + b2;
            } else {  // m >= r2
                result = k3 * m + b3;
            }
            if (result > upper_limit) result = upper_limit;
            temp.at<uchar>(r, c) = (uchar) result;
        }
    }

    temp.copyTo(dst);
}

void grayLayering(const cv::Mat &src, cv::Mat &dst, uint r1, uint r2, uint s, bool other_zero) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    // CV_8U -> [0, 255]
    const int upper_limit = 255;

    if (r1 > r2) {
        THROW_ARG_ERROR("Invalid `r1, r2`. You should make sure `r1 <= r2`.");
    }
    if (r2 > upper_limit) {
        THROW_ARG_ERROR("Invalid `r2`. You should make sure `r2 <= {}`.", upper_limit);
    }
    if (s > upper_limit) {
        THROW_ARG_ERROR("Invalid `s`. You should make sure `s <= {}`.", upper_limit);
    }

    // 系数,其他值置为 原值 / 零值
    int t = other_zero ? 0 : 1;

    cv::Mat temp(src.size(), CV_8UC1);
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            int m = src.at<uchar>(r, c);
            // 进行 灰度值级分层
            if (r1 <= m && m <= r2) {
                temp.at<uchar>(r, c) = (uchar) s;
            } else {  // m < r1 || m > r2
                temp.at<uchar>(r, c) = (uchar) (t * m);
            }
        }
    }

    temp.copyTo(dst);
}

void grayBitPlaneLayering(const cv::Mat &src, std::vector<cv::Mat> &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    // CV_8U
    const int bits = 8;

    dst.clear();
    for (int i = 0; i < bits; i++) {
        dst.emplace_back(cv::Mat::zeros(src.size(), CV_8UC1));
    }

    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            uchar m = src.at<uchar>(r, c);
            // 进行 比特平面分层
            for (int i = 0; i < bits; i++) {
                dst[i].at<uchar>(r, c) = (uchar) (m & (1 << i));
            }
        }
    }
}

cv::Mat grayHistogram(const cv::Mat &src, const cv::Mat &mask, cv::Size size,
                      const cv::Scalar &color) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    // 计算直方图 CV_8U -> 256
    int hist_size = 256;
    float range[] = {0, 256};
    const float *hist_range = {range};

    cv::Mat hist;
    cv::calcHist(&src, 1, nullptr, mask, hist, 1, &hist_size, &hist_range);

    // 归一化到 [0, 1]
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);

    // 绘制直方图
    cv::Mat hist_image_paint(size.height, size.width, CV_8UC3, cv::Scalar(0, 0, 0));
    int bin_width = cvRound((double) size.width / hist_size);
    for (int i = 1; i < hist_size; i++) {
        cv::Point p1 = cv::Point(bin_width * (i - 1),
                                 size.height - cvRound(hist.at<float>(i - 1) * (float) size.height));
        cv::Point p2 = cv::Point(bin_width * i,
                                 size.height - cvRound(hist.at<float>(i) * (float) size.height));
        cv::line(hist_image_paint, p1, p2, color, 2);
    }

    return hist_image_paint;
}

void localEqualizeHist(const cv::Mat &src, cv::Mat &dst, double clipLimit,
                       cv::Size tileGridSize) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileGridSize);
    clahe->apply(src, dst);
}

void matchHist(const cv::Mat &src, cv::Mat &dst, const cv::Mat &refer) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if (refer.empty()) {
        THROW_ARG_ERROR("Input refer image is empty.");
    }
    if (refer.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input refer image type must be CV_8UC1.");
    }

    // 对 原始图像 和 参考图像 进行直方图均衡
    cv::Mat src_equ, refer_equ;
    cv::equalizeHist(src, src_equ);
    cv::equalizeHist(refer, refer_equ);

    // 计算均衡后的图像的直方图 CV_8U -> 256
    const int hist_size = 256;
    float range[] = {0, hist_size};
    const float *hist_range = {range};

    cv::Mat src_hist, refer_hist;
    cv::calcHist(&src_equ, 1, nullptr, cv::Mat(), src_hist, 1, &hist_size, &hist_range);
    cv::calcHist(&refer_equ, 1, nullptr, cv::Mat(), refer_hist, 1, &hist_size, &hist_range);

    // 计算均衡后的图像的累计概率
    auto src_image_size = (float) (src_equ.rows * src_equ.cols);
    auto refer_image_size = (float) (refer_equ.rows * refer_equ.cols);

    float src_cdf[hist_size] = {(src_hist.at<float>(0) / src_image_size)};
    float refer_cdf[hist_size] = {(refer_hist.at<float>(0) / refer_image_size)};

    for (int i = 1; i < hist_size; i++) {
        src_cdf[i] = src_hist.at<float>(i) / src_image_size + src_cdf[i - 1];
        refer_cdf[i] = refer_hist.at<float>(i) / refer_image_size + refer_cdf[i - 1];
    }

    // 进行直方图规定化
    // 1. 计算累计概率的差值
    float diff_cdf[hist_size][hist_size];
    for (int i = 0; i < hist_size; i++) {
        for (int j = 0; j < hist_size; j++) {
            diff_cdf[i][j] = std::abs(src_cdf[i] - refer_cdf[j]);
        }
    }

    // 2. 构建灰度级映射表
    cv::Mat lut(1, hist_size, CV_8UC1);
    for (int i = 0; i < hist_size; i++) {
        // 查找累积概率差最小(灰度最接近)的规定化灰度
        float min_diff = diff_cdf[i][0];
        int index = 0;
        for (int j = 1; j < hist_size; j++) {
            if (diff_cdf[i][j] < min_diff) {
                min_diff = diff_cdf[i][j];
                index = j;
            }
        }
        lut.at<uchar>(i) = static_cast<uchar>(index);
    }

    // 3. 映射
    cv::Mat temp;
    cv::LUT(src_equ, lut, temp);

    temp.copyTo(dst);
}

void shadingCorrection(const cv::Mat &src, cv::Mat &dst, float k1, float k2) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if (k1 <= 0 || k1 > 0.5) {
        THROW_ARG_ERROR("Invalid `k1`: {}. You should make sure `0 < k1 <= 0.5`.", k1);
    }
    if (k2 <= 0 || k2 > 6) {
        THROW_ARG_ERROR("Invalid `k2`: {}. You should make sure `0 < k2 <= 6`.", k2);
    }

    // 计算卷积核参数
    cv::Size src_size = src.size();

    int ksize_width = (int) ((float) src_size.width * k1);
    if (ksize_width % 2 == 0) ksize_width += 1;

    int ksize_height = (int) ((float) src_size.height * k1);
    if (ksize_height % 2 == 0) ksize_height += 1;

    float sigma_x = (float) ksize_width / k2;
    float sigma_y = (float) ksize_height / k2;

    // 1. 通过高斯滤波获取阴影
    cv::Mat shading;
    cv::GaussianBlur(src, shading, cv::Size(ksize_width, ksize_height), sigma_x, sigma_y);

    // 2. 阴影校正
    cv::Mat temp(src_size, CV_32FC1);
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            temp.at<float>(r, c) = (float) src.at<uchar>(r, c) / (float) shading.at<uchar>(r, c);
        }
    }

    temp.copyTo(dst);
}
