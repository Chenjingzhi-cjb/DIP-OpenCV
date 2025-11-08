#include "spatial_filter.h"


void linearSpatialFilter(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (kernel.empty()) {
        THROW_ARG_ERROR("Input kernel is empty.");
    }

    // 初始化卷积核示例
    // cv::Mat kernel = (cv::Mat_<char>(3, 3) << 0, -1, 0, -1, 7, -1, 0, -1, 0);

    // 打印卷积核
    // std::cout << "default1\n" << kernel << std::endl;
    // std::cout << "default2\n" << cv::format(kernel, cv::Formatter::FMT_DEFAULT) << std::endl;
    // std::cout << "csv\n" << cv::format(kernel, cv::Formatter::FMT_CSV) << std::endl;
    // std::cout << "c\n" << cv::format(kernel, cv::Formatter::FMT_C) << std::endl;
    // std::cout << "numpy\n" << cv::format(kernel, cv::Formatter::FMT_NUMPY)  << std::endl;
    // std::cout << "matlab\n" << cv::format(kernel, cv::Formatter::FMT_MATLAB) << std::endl;
    // std::cout << "python\n" << cv::format(kernel, cv::Formatter::FMT_PYTHON) << std::endl;

    cv::filter2D(src, dst, src.depth(), kernel);
}

void smoothSpatialFilterBox(const cv::Mat &src, cv::Mat &dst, cv::Size ksize, cv::Point anchor, bool normalize,
                            int borderType) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    cv::boxFilter(src, dst, src.depth(), ksize, anchor, normalize, borderType);
}

void smoothSpatialFilterGauss(const cv::Mat &src, cv::Mat &dst, cv::Size ksize, double sigmaX, double sigmaY,
                              int borderType) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    cv::GaussianBlur(src, dst, ksize, sigmaX, sigmaY, borderType);
}

void orderStatisticsFilter(const cv::Mat &src, cv::Mat &dst, int ksize, int percentage) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if (ksize <= 0 || ksize % 2 == 0) {
        THROW_ARG_ERROR("Invalid `ksize`. You should make sure ksize is a positive odd number.");
    }
    if (percentage < 0 || percentage > 100) {
        THROW_ARG_ERROR("Invalid `percentage`. You should make sure `0 <= percentage <= 100`.");
    }

    // --- 中值滤波 ---
    if (percentage == 50) {
        cv::medianBlur(src, dst, ksize);
        return;
    }

    // --- 其他位置 ---
    int border = (ksize - 1) / 2;
    int index = ksize * ksize * percentage / 101;  // 101 避免数组越界

    cv::Mat src_copy;
    cv::copyMakeBorder(src, src_copy, border, border, border, border, cv::BORDER_REFLECT);

    cv::Mat temp(src_copy.size(), src_copy.type());
    std::vector<int> values{};

    for (int i = border; i < src_copy.rows - border; i++) {
        for (int j = border; j < src_copy.cols - border; j++) {
            values.clear();
            for (int k = -border; k <= border; k++) {
                for (int l = -border; l <= border; l++) {
                    values.emplace_back(src_copy.at<uchar>(i + k, j + l));
                }
            }
            std::sort(values.begin(), values.end());
            temp.at<uchar>(i, j) = (uchar) values[index];
        }
    }

    temp = temp(cv::Rect(border, border, src.cols, src.rows));

    temp.copyTo(dst);
}

void sharpenSpatialFilterLaplace(const cv::Mat &src, cv::Mat &dst, int ksize, double scale, double delta,
                                 int borderType) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    cv::Laplacian(src, dst, src.depth(), ksize, scale, delta, borderType);
}

void sharpenSpatialFilterTemplate(const cv::Mat &src, cv::Mat &dst, cv::Size smooth_ksize, float k) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (k < 0) {
        THROW_ARG_ERROR("Invalid `k`. You should make sure `k >= 0`.");
    }

    // 1. 获取模糊图像
    cv::Mat smooth_image(src.size(), src.type());
    cv::GaussianBlur(src, smooth_image, smooth_ksize, (smooth_ksize.width / 6.0), (smooth_ksize.height / 6.0));

    // 2. 原图像 - 模糊图像 = 模板
    cv::Mat template_image(src.size(), src.type());
    cv::subtract(src, smooth_image, template_image);

    // 3. 原图像 + k * 模板 = 结果图像
    cv::Mat result_image(src.size(), src.type());
    cv::addWeighted(src, 1, template_image, k, 0, result_image);

    result_image.copyTo(dst);
}

void sharpenSpatialFilterSobel(const cv::Mat &src, cv::Mat &dst, int dx, int dy, int ksize, double scale,
                               double delta, int borderType) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    cv::Sobel(src, dst, src.depth(), dx, dy, ksize, scale, delta, borderType);
}

void sharpenSpatialFilterScharr(const cv::Mat &src, cv::Mat &dst, int dx, int dy, double scale, double delta,
                                int borderType) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    cv::Scharr(src, dst, src.depth(), dx, dy, scale, delta, borderType);
}

void sharpenSpatialFilterCanny(const cv::Mat &src, cv::Mat &dst, double threshold1, double threshold2,
                               int apertureSize, bool L2gradient) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    cv::Canny(src, dst, threshold1, threshold2, apertureSize, L2gradient);
}

void geometricMeanFilter(const cv::Mat &src, cv::Mat &dst, cv::Size ksize) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if ((ksize.width <= 0) || (ksize.width % 2 == 0) || (ksize.height <= 0) || (ksize.height % 2 == 0)) {
        THROW_ARG_ERROR("Invalid `ksize`. You should make sure ksize is a positive odd number.");
    }

    int row_border = (ksize.height - 1) / 2;
    int col_border = (ksize.width - 1) / 2;

    cv::Mat src_copy;
    cv::copyMakeBorder(src, src_copy, row_border, row_border, col_border, col_border, cv::BORDER_REFLECT);
    src_copy.convertTo(src_copy, CV_64FC1, 1.0 / 255, 0);

    cv::Mat temp = cv::Mat::zeros(src_copy.size(), src_copy.type());

    for (int i = row_border; i < src_copy.rows - row_border; i++) {
        for (int j = col_border; j < src_copy.cols - col_border; j++) {
            double value = 1.0;
            for (int k = -row_border; k <= row_border; k++) {
                for (int l = -col_border; l <= col_border; l++) {
                    value *= src_copy.at<double>(i + k, j + l);
                }
            }
            temp.at<double>(i, j) = std::pow(value, 1.0 / (ksize.height * ksize.width));
        }
    }

    temp.convertTo(temp, CV_8UC1, 255, 0);
    cv::Mat temp_roi = temp(cv::Rect(col_border, row_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}

void harmonicAvgFilter(const cv::Mat &src, cv::Mat &dst, cv::Size ksize) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if ((ksize.width <= 0) || (ksize.width % 2 == 0) || (ksize.height <= 0) || (ksize.height % 2 == 0)) {
        THROW_ARG_ERROR("Invalid `ksize`. You should make sure ksize is a positive odd number.");
    }

    int row_border = (ksize.height - 1) / 2;
    int col_border = (ksize.width - 1) / 2;

    cv::Mat src_copy;
    cv::copyMakeBorder(src, src_copy, row_border, row_border, col_border, col_border, cv::BORDER_REFLECT);

    cv::Mat temp = cv::Mat::zeros(src_copy.size(), src_copy.type());

    for (int i = row_border; i < src_copy.rows - row_border; i++) {
        for (int j = col_border; j < src_copy.cols - col_border; j++) {
            double value = 0;
            for (int k = -row_border; k <= row_border; k++) {
                for (int l = -col_border; l <= col_border; l++) {
                    value += (1.0 / src_copy.at<uchar>(i + k, j + l));
                }
            }
            temp.at<uchar>(i, j) = (uchar) ((ksize.height * ksize.width) / value);
        }
    }

    cv::Mat temp_roi = temp(cv::Rect(col_border, row_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}

void antiHarmonicAvgFilter(const cv::Mat &src, cv::Mat &dst, cv::Size ksize, float order) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if ((ksize.width <= 0) || (ksize.width % 2 == 0) || (ksize.height <= 0) || (ksize.height % 2 == 0)) {
        THROW_ARG_ERROR("Invalid `ksize`. You should make sure ksize is a positive odd number.");
    }

    int row_border = (ksize.height - 1) / 2;
    int col_border = (ksize.width - 1) / 2;

    cv::Mat src_copy;
    cv::copyMakeBorder(src, src_copy, row_border, row_border, col_border, col_border, cv::BORDER_REFLECT);

    cv::Mat temp = cv::Mat::zeros(src_copy.size(), src_copy.type());

    for (int i = row_border; i < src_copy.rows - row_border; i++) {
        for (int j = col_border; j < src_copy.cols - col_border; j++) {
            double value_n = 0;  // 分子
            double value_d = 0;  // 分母
            for (int k = -row_border; k <= row_border; k++) {
                for (int l = -col_border; l <= col_border; l++) {
                    value_n += std::pow(src_copy.at<uchar>(i + k, j + l), order + 1);
                    value_d += std::pow(src_copy.at<uchar>(i + k, j + l), order);
                }
            }
            temp.at<uchar>(i, j) = (uchar) (value_n / value_d);
        }
    }

    cv::Mat temp_roi = temp(cv::Rect(col_border, row_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}

void midPointFilter(const cv::Mat &src, cv::Mat &dst, cv::Size ksize) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if ((ksize.width <= 0) || (ksize.width % 2 == 0) || (ksize.height <= 0) || (ksize.height % 2 == 0)) {
        THROW_ARG_ERROR("Invalid `ksize`. You should make sure ksize is a positive odd number.");
    }

    int row_border = (ksize.height - 1) / 2;
    int col_border = (ksize.width - 1) / 2;

    cv::Mat src_copy;
    cv::copyMakeBorder(src, src_copy, row_border, row_border, col_border, col_border, cv::BORDER_REFLECT);

    cv::Mat temp = cv::Mat::zeros(src_copy.size(), src_copy.type());
    std::vector<int> values{};

    for (int i = row_border; i < src_copy.rows - row_border; i++) {
        for (int j = col_border; j < src_copy.cols - col_border; j++) {
            values.clear();
            for (int k = -row_border; k <= row_border; k++) {
                for (int l = -col_border; l <= col_border; l++) {
                    values.emplace_back(src_copy.at<uchar>(i + k, j + l));
                }
            }
            auto min_max = std::minmax_element(values.begin(), values.end());
            temp.at<uchar>(i, j) = (uchar) ((*min_max.first + *min_max.second) / 2);
        }
    }

    cv::Mat temp_roi = temp(cv::Rect(col_border, row_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}

void modifiedAlphaMeanFilter(const cv::Mat &src, cv::Mat &dst, cv::Size ksize, int d) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if ((ksize.width <= 0) || (ksize.width % 2 == 0) || (ksize.height <= 0) || (ksize.height % 2 == 0)) {
        THROW_ARG_ERROR("Invalid `ksize`. You should make sure ksize is a positive odd number.");
    }
    if (d < 0 || d > (ksize.height * ksize.width - 1)) {
        THROW_ARG_ERROR("Invalid `d`. You should make sure `0 <= d <= mn-1`.");
    }

    int row_border = (ksize.height - 1) / 2;
    int col_border = (ksize.width - 1) / 2;

    cv::Mat src_copy;
    cv::copyMakeBorder(src, src_copy, row_border, row_border, col_border, col_border, cv::BORDER_REFLECT);

    int d_half = d / 2;
    cv::Mat temp = cv::Mat::zeros(src_copy.size(), src_copy.type());
    std::vector<int> values{};

    for (int i = row_border; i < src_copy.rows - row_border; i++) {
        for (int j = col_border; j < src_copy.cols - col_border; j++) {
            values.clear();
            for (int k = -row_border; k <= row_border; k++) {
                for (int l = -col_border; l <= col_border; l++) {
                    values.emplace_back(src_copy.at<uchar>(i + k, j + l));
                }
            }
            std::sort(values.begin(), values.end());
            temp.at<uchar>(i, j) = (uchar) (
                    std::accumulate(std::next(values.begin(), d_half), std::prev(values.end(), d_half), 0) /
                    (ksize.height * ksize.width - d_half * 2));
        }
    }

    cv::Mat temp_roi = temp(cv::Rect(col_border, row_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}

void adaptiveLocalFilter(const cv::Mat &src, cv::Mat &dst, cv::Size ksize) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if ((ksize.width <= 0) || (ksize.width % 2 == 0) || (ksize.height <= 0) || (ksize.height % 2 == 0)) {
        THROW_ARG_ERROR("Invalid `ksize`. You should make sure ksize is a positive odd number.");
    }

    cv::Mat src_mean, src_std;
    cv::meanStdDev(src, src_mean, src_std);
    double sigma_n = std::pow(src_std.at<double>(0), 2);  // 全局方差近似表示噪声方差

    int row_border = (ksize.height - 1) / 2;
    int col_border = (ksize.width - 1) / 2;

    cv::Mat src_copy;
    cv::copyMakeBorder(src, src_copy, row_border, row_border, col_border, col_border, cv::BORDER_REFLECT);

    cv::Mat temp = cv::Mat::zeros(src_copy.size(), src_copy.type());
    std::vector<int> values{};

    for (int i = row_border; i < src_copy.rows - row_border; i++) {
        for (int j = col_border; j < src_copy.cols - col_border; j++) {
            values.clear();
            for (int k = -row_border; k <= row_border; k++) {
                for (int l = -col_border; l <= col_border; l++) {
                    values.emplace_back(src_copy.at<uchar>(i + k, j + l));
                }
            }

            // 计算参数
            double mean_s = (double) std::accumulate(std::begin(values), std::end(values), 0) / (double) values.size();
            double sigma_s = 0;
            std::for_each(std::begin(values), std::end(values), [&](const int d) {
                sigma_s += std::pow(d - mean_s, 2);
            });
            sigma_s /= (double) values.size();
            int gxy = src_copy.at<uchar>(i, j);

            // 执行规则
            if (sigma_n <= sigma_s) {
                temp.at<uchar>(i, j) = (uchar) (gxy - (sigma_n / sigma_s) * (gxy - mean_s));
            } else {
                temp.at<uchar>(i, j) = (uchar) mean_s;
            }
        }
    }

    cv::Mat temp_roi = temp(cv::Rect(col_border, row_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}

void adaptiveMedianFilter(const cv::Mat &src, cv::Mat &dst, int max_ksize) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    int init_ksize = 3;
    int init_border = (init_ksize - 1) / 2;

    if ((max_ksize < init_ksize) || (max_ksize % 2 == 0)) {
        THROW_ARG_ERROR("Invalid `max_ksize`. You should make sure max_ksize is an odd number and `max_ksize >= 3`.");
    }

    cv::Mat src_copy;
    cv::copyMakeBorder(src, src_copy, init_border, init_border, init_border, init_border, cv::BORDER_REFLECT);

    cv::Mat temp = cv::Mat::zeros(src_copy.size(), src_copy.type());
    std::vector<int> values{};

    for (int i = init_border; i < src_copy.rows - init_border; i++) {
        for (int j = init_border; j < src_copy.cols - init_border; j++) {
            values.clear();
            int ksize = init_ksize;
            int border = init_border;
            while (true) {
                for (int k = -border; k <= border; k++) {
                    for (int l = -border; l <= border; l++) {
                        if ((i + k < 0) || (j + l < 0) || (i + k >= src_copy.rows) || (j + l >= src_copy.cols)) {
                            values.emplace_back(0);  // 零填充
                        }
                        values.emplace_back(src_copy.at<uchar>(i + k, j + l));
                    }
                }

                // 计算参数
                std::sort(values.begin(), values.end());
                int zmin = values.front();
                int zmax = values.back();
                int zmed = values[values.size() / 2];
                int zxy = src_copy.at<uchar>(i, j);

                // 执行规则
                if (zmin < zmed && zmed < zmax) {  // level B
                    if (zmin < zxy && zxy < zmax) {
                        temp.at<uchar>(i, j) = (uchar) zxy;
                    } else {
                        temp.at<uchar>(i, j) = (uchar) zmed;
                    }
                    break;
                } else {  // zmed == zmin || zmed == zmax, level A
                    ksize += 2;
                    border += 1;
                    if (ksize > max_ksize) {
                        temp.at<uchar>(i, j) = (uchar) zmed;
                        break;
                    }
                }
            }
        }
    }

    cv::Mat temp_roi = temp(cv::Rect(init_border, init_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}
