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

    Mat temp = Mat::zeros(src.size(), src.depth());

    filter2D(src, temp, src.depth(), kernel);

    temp.copyTo(dst);
}

void smoothSpatialFilterBox(Mat &src, Mat &dst, Size ksize, Point anchor, bool normalize, int borderType) {
    if (src.empty()) {
        throw invalid_argument("smoothSpatialFilterBox(): Input image is empty!");
    }

    Mat temp = Mat::zeros(src.size(), src.depth());

    boxFilter(src, temp, src.depth(), ksize, anchor, normalize, borderType);

    temp.copyTo(dst);
}

void smoothSpatialFilterGauss(Mat &src, Mat &dst, Size ksize, double sigmaX, double sigmaY, int borderType) {
    if (src.empty()) {
        throw invalid_argument("smoothSpatialFilterGauss(): Input image is empty!");
    }

    Mat temp = Mat::zeros(src.size(), src.depth());

    GaussianBlur(src, temp, ksize, sigmaX, sigmaY, borderType);

    temp.copyTo(dst);
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

    if (percentage < 0 || percentage > 100) {
        string err = R"(orderStatisticsFilter(): Parameter Error! You should make sure "0 <= percentage <= 100"!)";
        throw invalid_argument(err);
    }

    Mat temp = Mat::zeros(src.size(), src.depth());

    if (percentage == 50) {
        medianBlur(src, temp, ksize);
    } else {  // percentage != 50 TODO:
        cout << "orderStatisticsFilter(): Undeveloped, please look forward to! (Just too lazy to develop!)" << endl;
    }

    temp.copyTo(dst);
}

void sharpenSpatialFilterLaplace(Mat &src, Mat &dst, int ksize, double scale, double delta, int borderType) {
    if (src.empty()) {
        throw invalid_argument("sharpenSpatialFilterLaplace(): Input image is empty!");
    }

    Mat temp = Mat::zeros(src.size(), src.depth());

    Laplacian(src, temp, src.depth(), ksize, scale, delta, borderType);

    temp.copyTo(dst);
}

void sharpenSpatialFilterTemplate(Mat &src, Mat &dst, Size smooth_size, float k) {
    if (src.empty()) {
        throw invalid_argument("sharpenSpatialFilterTemplate(): Input image is empty!");
    }

    if (k < 0) {
        string err = R"(sharpenSpatialFilterTemplate(): Parameter Error! You should make sure "k >= 0"!)";
        throw invalid_argument(err);
    }

    // 1. 获取模糊图像
    Mat smooth_image = Mat::zeros(src.size(), src.depth());
    GaussianBlur(src, smooth_image, smooth_size, (smooth_size.width / 6.0), (smooth_size.height / 6.0));

    // 2. 原图像 - 模糊图像 = 模板
    Mat template_image = Mat::zeros(src.size(), src.depth());
    subtract(src, smooth_image, template_image);

    // 3. 原图像 + k * 模板 = 结果图像
    Mat temp = Mat::zeros(src.size(), src.depth());
    addWeighted(src, 1, template_image, k, 0, temp);

    temp.copyTo(dst);
}

void sharpenSpatialFilterSobel(Mat &src, Mat &dst, int dx, int dy, int ksize, double scale, double delta,
                               int borderType) {
    if (src.empty()) {
        throw invalid_argument("sharpenSpatialFilterSobel(): Input image is empty!");
    }

    Mat temp = Mat::zeros(src.size(), src.depth());

    Sobel(src, temp, src.depth(), dx, dy, ksize, scale, delta, borderType);

    temp.copyTo(dst);
}

void sharpenSpatialFilterScharr(Mat &src, Mat &dst, int dx, int dy, double scale, double delta, int borderType) {
    if (src.empty()) {
        throw invalid_argument("sharpenSpatialFilterScharr(): Input image is empty!");
    }

    Mat temp = Mat::zeros(src.size(), src.depth());

    Scharr(src, temp, src.depth(), dx, dy, scale, delta, borderType);

    temp.copyTo(dst);
}

void sharpenSpatialFilterCanny(Mat &src, Mat &dst, double threshold1, double threshold2, int apertureSize,
                               bool L2gradient) {
    if (src.empty()) {
        throw invalid_argument("sharpenSpatialFilterCanny(): Input image is empty!");
    }

    Mat temp = Mat::zeros(src.size(), src.depth());

    Canny(src, temp, threshold1, threshold2, apertureSize, L2gradient);

    temp.copyTo(dst);
}

