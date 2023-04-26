#include "spatial_filter.h"


void linearSpatialFilter(Mat &src, Mat &dst, Mat &kernel) {
    if (src.empty()) {
        throw invalid_argument("linearSpatialFilter(): Input image is empty!");
    }

    if (kernel.empty()) {
        throw invalid_argument("linearSpatialFilter(): Kernel is empty!");
    }

    // 初始化卷积核示例
    // Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 7, -1, 0, -1, 0);

    // 打印卷积核
    // cout << "csv\n" << format(kernel, Formatter::FMT_CSV) << endl;
    /* 其他方式
    cout << "default1\n" << kernel << endl;
    cout << "default2\n" << format(kernel, Formatter::FMT_DEFAULT) << endl;
    cout << "c\n" << format(kernel, Formatter::FMT_C) << endl;
    cout << "numpy\n" << format(kernel, Formatter::FMT_NUMPY)  << endl;
    cout << "matlab\n" << format(kernel, Formatter::FMT_MATLAB) << endl;
    cout << "python\n" << format(kernel, Formatter::FMT_PYTHON) << endl;
     */

    filter2D(src, dst, src.depth(), kernel);
}

void smoothSpatialFilterBox(Mat &src, Mat &dst, Size ksize, Point anchor, bool normalize, int borderType) {
    if (src.empty()) {
        throw invalid_argument("smoothSpatialFilterBox(): Input image is empty!");
    }

    boxFilter(src, dst, src.depth(), ksize, anchor, normalize, borderType);
}

void smoothSpatialFilterGauss(Mat &src, Mat &dst, Size ksize, double sigmaX, double sigmaY, int borderType) {
    if (src.empty()) {
        throw invalid_argument("smoothSpatialFilterGauss(): Input image is empty!");
    }

    GaussianBlur(src, dst, ksize, sigmaX, sigmaY, borderType);
}

void shadingCorrection(Mat &src, Mat &dst, float k1, float k2) {  // TODO:
    if (src.empty()) {
        throw invalid_argument("shadingCorrection(): Input image is empty!");
    }

    if (k1 <= 0 || k1 > 0.5) {
        string err = R"(shadingCorrection(): Parameter Error! You should make sure "0 < k <= 0.5"!)";
        throw invalid_argument(err);
    }

    namedWindow("src", WINDOW_AUTOSIZE);
    imshow("src", src);

    Size src_size = src.size();

    // 计算卷积核参数
    int ksize_width = (int) ((float) src_size.width * k1);
    if (ksize_width % 2 == 0) ksize_width += 1;

    int ksize_height = (int) ((float) src_size.height * k1);
    if (ksize_height % 2 == 0) ksize_height += 1;

    // 1. 通过高斯滤波获取阴影
    Mat shading = Mat::zeros(src_size, src.depth());
    GaussianBlur(src, shading, Size(ksize_width, ksize_height), (float) ksize_width / k2, (float) ksize_height / k2);

    namedWindow("shading", WINDOW_AUTOSIZE);
    imshow("shading", shading);

    // 2. 阴影校正
    Mat temp = Mat::zeros(src_size, src.depth());

    waitKey(0);
}

void orderStatisticsFilter(Mat &src, Mat &dst, int ksize, int percentage) {
    if (src.empty()) {
        throw invalid_argument("orderStatisticsFilter(): Input image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("orderStatisticsFilter(): Input image's type error! It should be CV_8UC1");
    }

    if (ksize <= 0 || ksize % 2 == 0) {
        string err = R"(orderStatisticsFilter(): Parameter Error! You should make sure ksize is a positive odd number!)";
        throw invalid_argument(err);
    }

    if (percentage < 0 || percentage > 100) {
        string err = R"(orderStatisticsFilter(): Parameter Error! You should make sure "0 <= percentage <= 100"!)";
        throw invalid_argument(err);
    }

    // 中值滤波器
    if (percentage == 50) {
        medianBlur(src, dst, ksize);
        return;
    }

    // 其他位置
    int border = (ksize - 1) / 2;

    Mat src_copy;
    copyMakeBorder(src, src_copy, border, border, border, border, BORDER_REFLECT);

    Mat temp = Mat::zeros(src_copy.size(), src_copy.type());
    int index = ksize * ksize * percentage / 101;  // 101 避免数组越界
    vector<int> values{};

    for (int i = border; i < src_copy.rows - border; i++) {
        for (int j = border; j < src_copy.cols - border; j++) {
            values.clear();
            for (int k = -border; k <= border; k++) {
                for (int l = -border; l <= border; l++) {
                    values.emplace_back(src_copy.at<uchar>(i + k, j + l));
                }
            }
            std::sort(values.begin(), values.end());
            temp.at<uchar>(i, j) = values[index];
        }
    }

    Mat temp_roi = temp(Rect(border, border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}

void sharpenSpatialFilterLaplace(Mat &src, Mat &dst, int ksize, double scale, double delta, int borderType) {
    if (src.empty()) {
        throw invalid_argument("sharpenSpatialFilterLaplace(): Input image is empty!");
    }

    Laplacian(src, dst, src.depth(), ksize, scale, delta, borderType);
}

void sharpenSpatialFilterTemplate(Mat &src, Mat &dst, Size smooth_ksize, float k) {
    if (src.empty()) {
        throw invalid_argument("sharpenSpatialFilterTemplate(): Input image is empty!");
    }

    if (k < 0) {
        string err = R"(sharpenSpatialFilterTemplate(): Parameter Error! You should make sure "k >= 0"!)";
        throw invalid_argument(err);
    }

    // 1. 获取模糊图像
    Mat smooth_image = Mat::zeros(src.size(), src.depth());
    GaussianBlur(src, smooth_image, smooth_ksize, (smooth_ksize.width / 6.0), (smooth_ksize.height / 6.0));

    // 2. 原图像 - 模糊图像 = 模板
    Mat template_image = Mat::zeros(src.size(), src.depth());
    subtract(src, smooth_image, template_image);

    // 3. 原图像 + k * 模板 = 结果图像
    Mat result_image = Mat::zeros(src.size(), src.depth());
    addWeighted(src, 1, template_image, k, 0, result_image);

    result_image.copyTo(dst);
}

void sharpenSpatialFilterSobel(Mat &src, Mat &dst, int dx, int dy, int ksize, double scale, double delta,
                               int borderType) {
    if (src.empty()) {
        throw invalid_argument("sharpenSpatialFilterSobel(): Input image is empty!");
    }

    Sobel(src, dst, src.depth(), dx, dy, ksize, scale, delta, borderType);
}

void sharpenSpatialFilterScharr(Mat &src, Mat &dst, int dx, int dy, double scale, double delta, int borderType) {
    if (src.empty()) {
        throw invalid_argument("sharpenSpatialFilterScharr(): Input image is empty!");
    }

    Scharr(src, dst, src.depth(), dx, dy, scale, delta, borderType);
}

void sharpenSpatialFilterCanny(Mat &src, Mat &dst, double threshold1, double threshold2, int apertureSize,
                               bool L2gradient) {
    if (src.empty()) {
        throw invalid_argument("sharpenSpatialFilterCanny(): Input image is empty!");
    }

    Canny(src, dst, threshold1, threshold2, apertureSize, L2gradient);
}

void geometricMeanFilter(Mat &src, Mat &dst, Size ksize) {
    if (src.empty()) {
        throw invalid_argument("geometricMeanFilter(): Input image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("geometricMeanFilter(): Input image's type error! It should be CV_8UC1");
    }

    if ((ksize.width <= 0) || (ksize.width % 2 == 0) || (ksize.height <= 0) || (ksize.height % 2 == 0)) {
        string err = R"(geometricMeanFilter(): Parameter Error! You should make sure ksize is positive odd number!)";
        throw invalid_argument(err);
    }

    int row_border = (ksize.height - 1) / 2;
    int col_border = (ksize.width - 1) / 2;

    Mat src_copy;
    copyMakeBorder(src, src_copy, row_border, row_border, col_border, col_border, BORDER_REFLECT);
    src_copy.convertTo(src_copy, CV_64FC1, 1.0 / 255, 0);

    Mat temp = Mat::zeros(src_copy.size(), src_copy.type());

    for (int i = row_border; i < src_copy.rows - row_border; i++) {
        for (int j = col_border; j < src_copy.cols - col_border; j++) {
            double value = 1.0;
            for (int k = -row_border; k <= row_border; k++) {
                for (int l = -col_border; l <= col_border; l++) {
                    value *= src_copy.at<double>(i + k, j + l);
                }
            }
            temp.at<double>(i, j) = pow(value, 1.0 / (ksize.height * ksize.width));
        }
    }

    temp.convertTo(temp, CV_8UC1, 255, 0);
    Mat temp_roi = temp(Rect(col_border, row_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}

void harmonicAvgFilter(Mat &src, Mat &dst, Size ksize) {
    if (src.empty()) {
        throw invalid_argument("harmonicAvgFilter(): Input image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("harmonicAvgFilter(): Input image's type error! It should be CV_8UC1");
    }

    if ((ksize.width <= 0) || (ksize.width % 2 == 0) || (ksize.height <= 0) || (ksize.height % 2 == 0)) {
        string err = R"(harmonicAvgFilter(): Parameter Error! You should make sure ksize is positive odd number!)";
        throw invalid_argument(err);
    }

    int row_border = (ksize.height - 1) / 2;
    int col_border = (ksize.width - 1) / 2;

    Mat src_copy;
    copyMakeBorder(src, src_copy, row_border, row_border, col_border, col_border, BORDER_REFLECT);

    Mat temp = Mat::zeros(src_copy.size(), src_copy.type());

    for (int i = row_border; i < src_copy.rows - row_border; i++) {
        for (int j = col_border; j < src_copy.cols - col_border; j++) {
            double value = 0;
            for (int k = -row_border; k <= row_border; k++) {
                for (int l = -col_border; l <= col_border; l++) {
                    value += (1.0 / src_copy.at<uchar>(i + k, j + l));
                }
            }
            temp.at<uchar>(i, j) = (int) ((ksize.height * ksize.width) / value);
        }
    }

    Mat temp_roi = temp(Rect(col_border, row_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}

void antiHarmonicAvgFilter(Mat &src, Mat &dst, Size ksize, float order) {
    if (src.empty()) {
        throw invalid_argument("antiHarmonicAvgFilter(): Input image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("antiHarmonicAvgFilter(): Input image's type error! It should be CV_8UC1");
    }

    if ((ksize.width <= 0) || (ksize.width % 2 == 0) || (ksize.height <= 0) || (ksize.height % 2 == 0)) {
        string err = R"(antiHarmonicAvgFilter(): Parameter Error! You should make sure ksize is positive odd number!)";
        throw invalid_argument(err);
    }

    int row_border = (ksize.height - 1) / 2;
    int col_border = (ksize.width - 1) / 2;

    Mat src_copy;
    copyMakeBorder(src, src_copy, row_border, row_border, col_border, col_border, BORDER_REFLECT);

    Mat temp = Mat::zeros(src_copy.size(), src_copy.type());

    for (int i = row_border; i < src_copy.rows - row_border; i++) {
        for (int j = col_border; j < src_copy.cols - col_border; j++) {
            double value_n = 0;  // 分子
            double value_d = 0;  // 分母
            for (int k = -row_border; k <= row_border; k++) {
                for (int l = -col_border; l <= col_border; l++) {
                    value_n += pow(src_copy.at<uchar>(i + k, j + l), order + 1);
                    value_d += pow(src_copy.at<uchar>(i + k, j + l), order);
                }
            }
            temp.at<uchar>(i, j) = (int) (value_n / value_d);
        }
    }

    Mat temp_roi = temp(Rect(col_border, row_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}

void midPointFilter(Mat &src, Mat &dst, Size ksize) {
    if (src.empty()) {
        throw invalid_argument("midPointFilter(): Input image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("midPointFilter(): Input image's type error! It should be CV_8UC1");
    }

    if ((ksize.width <= 0) || (ksize.width % 2 == 0) || (ksize.height <= 0) || (ksize.height % 2 == 0)) {
        string err = R"(midPointFilter(): Parameter Error! You should make sure ksize is positive odd number!)";
        throw invalid_argument(err);
    }

    int row_border = (ksize.height - 1) / 2;
    int col_border = (ksize.width - 1) / 2;

    Mat src_copy;
    copyMakeBorder(src, src_copy, row_border, row_border, col_border, col_border, BORDER_REFLECT);

    Mat temp = Mat::zeros(src_copy.size(), src_copy.type());
    vector<int> values{};

    for (int i = row_border; i < src_copy.rows - row_border; i++) {
        for (int j = col_border; j < src_copy.cols - col_border; j++) {
            values.clear();
            for (int k = -row_border; k <= row_border; k++) {
                for (int l = -col_border; l <= col_border; l++) {
                    values.emplace_back(src_copy.at<uchar>(i + k, j + l));
                }
            }
            auto min_max = minmax_element(values.begin(), values.end());
            temp.at<uchar>(i, j) = (*min_max.first + *min_max.second) / 2;
        }
    }

    Mat temp_roi = temp(Rect(col_border, row_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}

void modifiedAlphaMeanFilter(Mat &src, Mat &dst, Size ksize, int d) {
    if (src.empty()) {
        throw invalid_argument("modifiedAlphaMeanFilter(): Input image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("modifiedAlphaMeanFilter(): Input image's type error! It should be CV_8UC1");
    }

    if ((ksize.width <= 0) || (ksize.width % 2 == 0) || (ksize.height <= 0) || (ksize.height % 2 == 0)) {
        string err = R"(modifiedAlphaMeanFilter(): Parameter Error! You should make sure ksize is positive odd number!)";
        throw invalid_argument(err);
    }

    if (d < 0 || d > (ksize.height * ksize.width - 1)) {
        string err = R"(modifiedAlphaMeanFilter(): Parameter Error! You should make sure "0 <= d <= mn-1"!)";
        throw invalid_argument(err);
    }

    int row_border = (ksize.height - 1) / 2;
    int col_border = (ksize.width - 1) / 2;

    Mat src_copy;
    copyMakeBorder(src, src_copy, row_border, row_border, col_border, col_border, BORDER_REFLECT);

    int d_half = d / 2;
    Mat temp = Mat::zeros(src_copy.size(), src_copy.type());
    vector<int> values{};

    for (int i = row_border; i < src_copy.rows - row_border; i++) {
        for (int j = col_border; j < src_copy.cols - col_border; j++) {
            values.clear();
            for (int k = -row_border; k <= row_border; k++) {
                for (int l = -col_border; l <= col_border; l++) {
                    values.emplace_back(src_copy.at<uchar>(i + k, j + l));
                }
            }
            std::sort(values.begin(), values.end());
            temp.at<uchar>(i, j) = accumulate(next(values.begin(), d_half), prev(values.end(), d_half), 0) /
                                   (ksize.height * ksize.width - d_half * 2);
        }
    }

    Mat temp_roi = temp(Rect(col_border, row_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}

void adaptiveLocalFilter(Mat &src, Mat &dst, Size ksize) {
    if (src.empty()) {
        throw invalid_argument("adaptiveLocalFilter(): Input image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("adaptiveLocalFilter(): Input image's type error! It should be CV_8UC1");
    }

    if ((ksize.width <= 0) || (ksize.width % 2 == 0) || (ksize.height <= 0) || (ksize.height % 2 == 0)) {
        string err = R"(adaptiveLocalFilter(): Parameter Error! You should make sure ksize is positive odd number!)";
        throw invalid_argument(err);
    }

    Mat src_mean, src_std;
    meanStdDev(src, src_mean, src_std);
    double sigma_n = pow(src_std.at<double>(0), 2);  // 全局方差近似表示噪声方差

    int row_border = (ksize.height - 1) / 2;
    int col_border = (ksize.width - 1) / 2;

    Mat src_copy;
    copyMakeBorder(src, src_copy, row_border, row_border, col_border, col_border, BORDER_REFLECT);

    Mat temp = Mat::zeros(src_copy.size(), src_copy.type());
    vector<int> values{};

    for (int i = row_border; i < src_copy.rows - row_border; i++) {
        for (int j = col_border; j < src_copy.cols - col_border; j++) {
            values.clear();
            for (int k = -row_border; k <= row_border; k++) {
                for (int l = -col_border; l <= col_border; l++) {
                    values.emplace_back(src_copy.at<uchar>(i + k, j + l));
                }
            }

            // 计算参数
            double mean_s = (double) accumulate(begin(values), end(values), 0) / (double) values.size();
            double sigma_s  = 0;
            for_each (begin(values), end(values), [&](const int d) {
                sigma_s += pow(d - mean_s, 2);
            });
            sigma_s /= (double) values.size();
            int gxy = src_copy.at<uchar>(i, j);

            // 执行规则
            if (sigma_n <= sigma_s) {
                temp.at<uchar>(i, j) = (int) (gxy - (sigma_n / sigma_s) * (gxy - mean_s));
            } else {
                temp.at<uchar>(i, j) = (int) mean_s;
            }
        }
    }

    Mat temp_roi = temp(Rect(col_border, row_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}

void adaptiveMedianFilter(Mat &src, Mat &dst, int max_ksize) {
    if (src.empty()) {
        throw invalid_argument("adaptiveMedianFilter(): Input image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("adaptiveMedianFilter(): Input image's type error! It should be CV_8UC1");
    }

    int init_ksize = 3;
    int init_border = (init_ksize - 1) / 2;

    if ((max_ksize < init_ksize) || (max_ksize % 2 == 0)) {
        string err = R"(adaptiveMedianFilter(): Parameter Error! You should make sure max_ksize is a odd number and "max_ksize >= 3"!)";
        throw invalid_argument(err);
    }

    Mat src_copy;
    copyMakeBorder(src, src_copy, init_border, init_border, init_border, init_border, BORDER_REFLECT);

    Mat temp = Mat::zeros(src_copy.size(), src_copy.type());
    vector<int> values{};

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
                        temp.at<uchar>(i, j) = zxy;
                    } else {
                        temp.at<uchar>(i, j) = zmed;
                    }
                    break;
                } else {  // zmed == zmin || zmed == zmax, level A
                    ksize += 2;
                    border += 1;
                    if (ksize > max_ksize) {
                        temp.at<uchar>(i, j) = zmed;
                        break;
                    }
                }
            }
        }
    }

    Mat temp_roi = temp(Rect(init_border, init_border, src.cols, src.rows));

    temp_roi.copyTo(dst);
}
