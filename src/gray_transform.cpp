#include "gray_transform.h"


void bgrToGray(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("bgrToGray(): Input image is empty!");
    }

    Mat temp = Mat::zeros(src.size(), src.depth());

    cvtColor(src, temp, COLOR_BGR2GRAY);

    temp.copyTo(dst);
}

void grayLinearScaleCV_8U(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("grayLinearScaleCV_8U(): Input image is empty!");
    }

    // CV_8U -> [0-255]
    int upper_limit = 255;

    double min_value, max_value;
    Point min_idx, max_idx;
    minMaxLoc(src, &min_value, &max_value, &min_idx, &max_idx);
    if ((max_value - min_value) == 0) return;
    double k = upper_limit / (max_value - min_value);

    Mat temp = Mat::zeros(src.size(), src.depth());
    src.convertTo(temp, src.depth(), k, -1 * k * min_value);

    temp.copyTo(dst);
}

void grayInvert(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("grayInvert(): Input image is empty!");
    }

    // CV_8U -> [0-255]
    int upper_limit = 255;

    Mat temp = Mat::zeros(src.size(), src.depth());
    src.convertTo(temp, src.depth(), -1, upper_limit);

    temp.copyTo(dst);
}

void grayLog(Mat &src, Mat &dst, float k) {
    if (src.empty()) {
        throw invalid_argument("grayLog(): Input image is empty!");
    }

    // CV_8U -> [0-255]
    int upper_limit = 255;

    double max_value = k * log10(1 + upper_limit);
    if (max_value == 0) return;
    double d = upper_limit / max_value;

    Mat temp = Mat::zeros(src.size(), src.depth());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            int m = src.at<uchar>(r, c);
            // 进行 对数变换 并 线性缩放至 [0-255]
            temp.at<uchar>(r, c) = uchar(k * log10(1 + m) * d);
        }
    }

    temp.copyTo(dst);
}

void grayAntiLog(Mat &src, Mat &dst, float k) {
    if (src.empty()) {
        throw invalid_argument("grayAntiLog(): Input image is empty!");
    }

    // CV_8U -> [0-255]
    int upper_limit = 255;

    double max_value = k * (pow(10, upper_limit) - 1);
    if (max_value == 0) return;
    double d = upper_limit / max_value;

    Mat temp = Mat::zeros(src.size(), src.depth());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            int m = src.at<uchar>(r, c);
            // 进行 反对数变换 并 线性缩放至 [0-255]
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
    
    // CV_8U -> [0-255]
    int upper_limit = 255;
    
    double max_value = k * pow(upper_limit, gamma);
    if (max_value == 0) return;
    double d = upper_limit / max_value;

    Mat temp = Mat::zeros(src.size(), src.depth());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            int m = src.at<uchar>(r, c);
            // 进行 伽马变换 并 线性缩放至 [0-255]
            temp.at<uchar>(r, c) = uchar(k * pow(m, gamma) * d);
        }
    }

    temp.copyTo(dst);
}

void grayContrastStretch(Mat &src, Mat &dst, uint r1, uint s1, uint r2, uint s2) {
    if (src.empty()) {
        throw invalid_argument("grayContrastStretch(): Input image is empty!");
    }

    // CV_8U -> [0-255]
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

    // CV_8U -> [0-255]
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

