#include "gray_transform.h"


void bgrToGray(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("bgrToGray(): Input image is empty!");
    }

    cvtColor(src, dst, COLOR_BGR2GRAY);
}

void grayLinearScaleCV_8U(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("grayLinearScaleCV_8U(): Input image is empty!");
    }

    normalize(src, dst, 0, 255, NORM_MINMAX, CV_8U);
}

void grayInvert(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("grayInvert(): Input image is empty!");
    }

    src.convertTo(dst, CV_8U, -1, 255);
}

void grayLog(Mat &src, Mat &dst, float k) {
    if (src.empty()) {
        throw invalid_argument("grayLog(): Input image is empty!");
    }

    // CV_8U -> [0, 255]
    int upper_limit = 255;

    double max_value = k * log10(1 + upper_limit);
    if (max_value == 0) return;
    double d = upper_limit / max_value;

    Mat temp = Mat::zeros(src.size(), src.depth());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            int m = src.at<uchar>(r, c);
            // 进行 对数变换 并 线性缩放至 [0, 255]
            temp.at<uchar>(r, c) = uchar(k * log10(1 + m) * d);
        }
    }

    temp.copyTo(dst);
}

void grayAntiLog(Mat &src, Mat &dst, float k) {
    if (src.empty()) {
        throw invalid_argument("grayAntiLog(): Input image is empty!");
    }

    // CV_8U -> [0, 255]
    int upper_limit = 255;

    double max_value = k * (pow(10, upper_limit) - 1);
    if (max_value == 0) return;
    double d = upper_limit / max_value;

    Mat temp = Mat::zeros(src.size(), src.depth());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            int m = src.at<uchar>(r, c);
            // 进行 反对数变换 并 线性缩放至 [0, 255]
            temp.at<uchar>(r, c) = uchar(k * (pow(10, m) - 1) * d);
        }
    }

    temp.copyTo(dst);
}

void grayGamma(Mat &src, Mat &dst, float k, float gamma) {
    if (src.empty()) {
        throw invalid_argument("grayGamma(): Input image is empty!");
    }

    if (gamma <= 0) {
        string err = R"(grayGamma(): Parameter Error! You should make sure "gamma > 0"!)";
        throw invalid_argument(err);
    }

    // CV_8U -> [0, 255]
    int upper_limit = 255;

    double max_value = k * pow(upper_limit, gamma);
    if (max_value == 0) return;
    double d = upper_limit / max_value;

    Mat temp = Mat::zeros(src.size(), src.depth());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            int m = src.at<uchar>(r, c);
            // 进行 伽马变换 并 线性缩放至 [0, 255]
            temp.at<uchar>(r, c) = uchar(k * pow(m, gamma) * d);
        }
    }

    temp.copyTo(dst);
}

void grayContrastStretch(Mat &src, Mat &dst, uint r1, uint s1, uint r2, uint s2) {
    if (src.empty()) {
        throw invalid_argument("grayContrastStretch(): Input image is empty!");
    }

    // CV_8U -> [0, 255]
    int upper_limit = 255;

    if (r1 >= r2 || s1 >= s2) {
        string err = "grayContrastStretch(): Parameter Error! ";
        err += R"(You should make sure "r1 < r2" and "s1 < s2"!)";
        throw invalid_argument(err);
    }

    if (r1 <= 0 || s1 <= 0) {
        string err = "grayContrastStretch(): Parameter Error! ";
        err += R"(You should make sure "r1, s1 > 0"!)";
        throw invalid_argument(err);
    }

    if (r2 >= upper_limit || s2 >= upper_limit) {
        string err = "grayContrastStretch(): Parameter Error! ";
        err += R"(You should make sure "r2, s2 < )";
        err += to_string(upper_limit);
        err += R"("!)";
        throw invalid_argument(err);
    }

    double k1 = (double) s1 / r1;
    double k2 = (double) (s2 - s1) / (r2 - r1);
    double b2 = s1 - k2 * r1;
    double k3 = (double) (upper_limit - s2) / (upper_limit - r2);
    double b3 = s2 - k3 * r2;

    Mat temp = Mat::zeros(src.size(), src.depth());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            int m = src.at<uchar>(r, c);
            // 进行 对比度拉伸
            double result;
            if (m < r1) {
                result = k1 * m;
            } else if (r1 <= m && m < r2) {
                result = k2 * m + b2;
            } else {  // m >= r2
                result = k3 * m + b3;
            }
            if (result > upper_limit) result = upper_limit;
            temp.at<uchar>(r, c) = uchar(result);
        }
    }

    temp.copyTo(dst);
}

void grayLayering(Mat &src, Mat &dst, uint r1, uint r2, uint s, bool other_zero) {
    if (src.empty()) {
        throw invalid_argument("grayLayering(): Input image is empty!");
    }

    // CV_8U -> [0, 255]
    int upper_limit = 255;

    if (r1 > r2) {
        throw invalid_argument(R"(grayLayering(): Parameter Error! You should make sure "r1 <= r2"!)");
    }

    if (r1 < 0) {
        throw invalid_argument(R"(grayLayering(): Parameter Error! You should make sure "r1 >= 0"!)");
    }

    if (r2 > upper_limit) {
        string err = "grayLayering(): Parameter Error! ";
        err += R"(You should make sure "r2 <= )";
        err += to_string(upper_limit);
        err += R"("!)";
        throw invalid_argument(err);
    }

    if (s < 0 || s > upper_limit) {
        string err = "grayLayering(): Parameter Error! ";
        err += R"(You should make sure "s >= 0" and "s <= )";
        err += to_string(upper_limit);
        err += R"("!)";
        throw invalid_argument(err);
    }

    // 系数，其他值置为 原值 / 零值
    int t = 1;
    if (other_zero) t = 0;

    Mat temp = Mat::zeros(src.size(), src.depth());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            int m = src.at<uchar>(r, c);
            // 进行 灰度值级分层
            if (r1 <= m && m <= r2) {
                temp.at<uchar>(r, c) = s;
            } else {  // m < r1 || m > r2
                temp.at<uchar>(r, c) = uchar(t * m);
            }
        }
    }

    temp.copyTo(dst);
}

void grayBitPlaneLayering(Mat &src, vector<Mat> &dst) {
    if (src.empty()) {
        throw invalid_argument("grayBitPlaneLayering(): Input image is empty!");
    }

    // CV_8U
    int bits = 8;

    dst.clear();
    for (int i = 0; i < bits; i++) {
        dst.emplace_back(Mat::zeros(src.size(), src.depth()));
    }

    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            uchar m = src.at<uchar>(r, c);
            // 进行 比特平面分层
            for (int i = 0; i < bits; i++) {
                dst[i].at<uchar>(r, c) = uchar(m & (1 << i));
            }
        }
    }
}

Mat grayHistogram(Mat &src, Size size, const Scalar &color) {
    if (src.empty()) {
        throw invalid_argument("grayHistogram(): Input image is empty!");
    }

    // 计算直方图 CV_8U -> 256
    Mat hist_image_cal;
    int hist_size = 256;
    float range[] = {0, 256};
    const float *hist_range = {range};
    calcHist(&src, 1, nullptr, Mat(), hist_image_cal, 1, &hist_size, &hist_range);

    // 归一化到 [0, 1]
    normalize(hist_image_cal, hist_image_cal, 0, 1, NORM_MINMAX);

    // 绘制直方图
    Mat hist_image_paint(size.height, size.width, CV_8UC3, Scalar(0, 0, 0));
    int bin_width = cvRound((double) size.width / hist_size);
    for (int i = 1; i < hist_size; i++) {
        line(hist_image_paint,
             Point(bin_width * (i - 1), size.height - cvRound(hist_image_cal.at<float>(i - 1) * (float) size.height)),
             Point(bin_width * i, size.height - cvRound(hist_image_cal.at<float>(i) * (float) size.height)),
             color, 2);
    }

    return hist_image_paint;
}

void localEqualizeHist(Mat &src, Mat &dst, double clipLimit, Size tileGridSize) {
    if (src.empty()) {
        throw invalid_argument("localEqualizeHist(): Input src image is empty!");
    }
    
    Ptr<CLAHE> clahe = createCLAHE(clipLimit, tileGridSize);
    clahe->apply(src, dst);
}

void matchHist(Mat &src, Mat &dst, Mat &refer) {
    if (src.empty()) {
        throw invalid_argument("matchHist(): Input src image is empty!");
    }

    if (refer.empty()) {
        throw invalid_argument("matchHist(): Input refer image is empty!");
    }

    // 对 原始图像 和 参考图像 进行直方图均衡
    Mat src_equ, refer_equ;
    equalizeHist(src, src_equ);
    equalizeHist(refer, refer_equ);

    // 计算均衡后的图像的直方图 CV_8U -> 256
    Mat src_hist, refer_hist;
    const int hist_size = 256;
    float range[] = {0, hist_size};
    const float *hist_range = {range};
    calcHist(&src_equ, 1, nullptr, Mat(), src_hist, 1, &hist_size, &hist_range);
    calcHist(&refer_equ, 1, nullptr, Mat(), refer_hist, 1, &hist_size, &hist_range);

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
    Mat matched = Mat::zeros(src.size(), CV_8UC1);
    // 1. 计算累计概率的差值
    float diff_cdf[hist_size][hist_size];
    for (int i = 0; i < hist_size; i++) {
        for (int j = 0; j < hist_size; j++) {
            diff_cdf[i][j] = fabs(src_cdf[i] - refer_cdf[j]);
        }
    }
    // 2. 构建灰度级映射表
    Mat lut(1, hist_size, CV_8UC1);
    for (int i = 1; i < hist_size; i++) {
        // 查找累积概率差最小(灰度最接近)的规定化灰度
        float min = diff_cdf[i][0];
        int index = 0;
        for (int j = 0; j < hist_size; j++) {
            if (diff_cdf[i][j] < min) {
                min = diff_cdf[i][j];
                index = j;
            }
        }
        lut.at<uchar>(i) = index;
    }
    // 3. 映射
    LUT(src_equ, lut, matched);

    matched.copyTo(dst);
}
