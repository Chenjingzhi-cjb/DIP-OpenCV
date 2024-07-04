#include "other_application.h"


pair<double, double> calcImageOffset(Mat &image_tmpl, Mat &image_offset) {
    if (image_tmpl.empty()) {
        throw invalid_argument("calcImageOffset() Error: Unable to load image_tmpl or image_tmpl loading error!");
    }

    if (image_offset.empty()) {
        throw invalid_argument("calcImageOffset() Error: Unable to load image_offset or image_offset loading error!");
    }

    Mat tmpl_float, offset_float;
    image_tmpl.convertTo(tmpl_float, CV_32FC1);
    image_offset.convertTo(offset_float, CV_32FC1);

    int tmpl_width = tmpl_float.cols;
    int tmpl_height = tmpl_float.rows;
    int offset_width = offset_float.cols;
    int offset_height = offset_float.rows;

    // 进行模板匹配
    Mat temp;
    matchTemplate(offset_float, tmpl_float, temp, TM_CCOEFF_NORMED);

    // 定位匹配的位置
    double max_val;  // 匹配得分，用于判断是否成功匹配
    Point max_loc;
    minMaxLoc(temp, nullptr, &max_val, nullptr, &max_loc);

    // 使用亚像素级的插值来精确定位模板的中心
    Point2f subpixel_offset = phaseCorrelate(tmpl_float,
                                             offset_float(Rect(max_loc.x, max_loc.y, tmpl_width, tmpl_height)));

    // 返回结果，中心坐标系像素
    pair<double, double> offset_value;
    offset_value.first = ((double) max_loc.x + subpixel_offset.x) - (int) ((offset_width - tmpl_width) / 2);
    offset_value.second = (int) ((offset_height - tmpl_height) / 2) - ((double) max_loc.y + subpixel_offset.y);

    return offset_value;
}

pair<double, double> calcImageOffset(Mat &image_std, Mat &image_offset, double tmpl_divisor) {
    if (image_std.empty()) {
        throw invalid_argument("calcImageOffset() Error: Unable to load image_std or image_std loading error!");
    }

    if (image_offset.empty()) {
        throw invalid_argument("calcImageOffset() Error: Unable to load image_offset or image_offset loading error!");
    }

    if (tmpl_divisor <= 1) {
        throw invalid_argument("calcImageOffset() Error: Parameter tmpl_divisor must be > 1!");
    }

    int std_width = image_std.cols;
    int std_height = image_std.rows;

    // 从标准图像中截取中心模板
    int tmpl_width = (int) (std_width / tmpl_divisor);
    int tmpl_height = (int) (std_height / tmpl_divisor);
    Mat image_tmpl = image_std(
            Rect((std_width - tmpl_width) / 2, (std_height - tmpl_height) / 2, tmpl_width, tmpl_height));

    // 计算图像偏移
    return calcImageOffset(image_tmpl, image_offset);
}

// 计算图像的清晰度值（梯度方法）
double calcSharpnessGradientScore(const Mat &image) {
    Mat grad_x, grad_y;
    Sobel(image, grad_x, CV_64F, 1, 0);
    Sobel(image, grad_y, CV_64F, 0, 1);
    Mat magnitude, angle;
    cartToPolar(grad_x, grad_y, magnitude, angle);

    Scalar mean_magnitude = mean(magnitude);
    return mean_magnitude[0];
}

// 计算图像的清晰度值（方差方法）
double calcSharpnessVarianceScore(const Mat &image) {
    Mat mean, stddev;
    meanStdDev(image, mean, stddev);
    return stddev.at<double>(0);
}

// 计算图像的清晰度值（频谱方法）
double calcSharpnessSpectrumScore(const Mat &image) {
    Mat complex_image;
    image.convertTo(complex_image, CV_32F);
    dft(complex_image, complex_image);

    // 分离复数图像到实部和虚部
    Mat planes[2];
    split(complex_image, planes);

    // 计算幅度谱
    Mat magnitude_image;
    magnitude(planes[0], planes[1], magnitude_image);

    Scalar mean_magnitude = mean(magnitude_image);
    return mean_magnitude[0];
}

// 计算图像的清晰度值（能量方法）
double calcSharpnessEnergyScore(const Mat &image) {
    Mat squared_image;
    multiply(image, image, squared_image);
    Scalar sum_squared_image = sum(squared_image);

    return sum_squared_image[0];
}

double calcSharpnessOld(cv::Mat *img) {
    std::vector<cv::Point> points;

    // get points
    for (int i = 3; i < img->rows - 3; i++) {
        int row_max_pixel = 0;
        int row_min_pixel = 255;

        auto *row = img->ptr<uchar>(i);
        for (int j = 3; j < img->cols - 3; j++) {
            if (row[j] > row_max_pixel) {
                row_max_pixel = row[j];
                points.emplace_back(cv::Point(i, j));
            }
            if (row[j] < row_min_pixel) {
                row_min_pixel = row[j];
                points.emplace_back(cv::Point(i, j));
            }
        }
    }

    // calculate value
    double S_vertical = 0, S_horizontal = 0;
    for (auto &i : points) {
        double vertical_gradient, horizontal_gradient;
        int col_no = i.y;
        int row_no = i.x;

        auto *plus_last_row = img->ptr<uchar>(row_no - 2);
        auto *last_row = img->ptr<uchar>(row_no - 1);
        auto *current_row = img->ptr<uchar>(row_no);
        auto *next_row = img->ptr<uchar>(row_no + 1);
        auto *plus_next_row = img->ptr<uchar>(row_no + 2);

        vertical_gradient =
                -plus_last_row[col_no - 2] - plus_last_row[col_no + 2] - last_row[col_no - 1] - last_row[col_no + 1] -
                last_row[col_no] + next_row[col_no - 1] + next_row[col_no] + next_row[col_no + 1] +
                plus_next_row[col_no - 2] + plus_next_row[col_no + 2];
        horizontal_gradient =
                -plus_last_row[col_no - 2] + plus_last_row[col_no + 2] - last_row[col_no - 1] + last_row[col_no + 1] -
                current_row[col_no - 1] + current_row[col_no + 1] - next_row[col_no - 1] + next_row[col_no + 1] -
                plus_next_row[col_no - 2] + plus_next_row[col_no + 2];

        S_vertical += fabs(vertical_gradient);
        S_horizontal += fabs(horizontal_gradient);
    }
    double S = S_vertical / (int) points.size() + S_horizontal / (int) points.size();

    return S;
}

double calcImageSharpness(Mat &image) {
    if (image.empty()) {
        throw invalid_argument("calcImageSharpness(): Input image is empty!");
    }

    double gradient_score = calcSharpnessGradientScore(image);  // 梯度方法
//    double variance_score = calcSharpnessVarianceScore(image);  // 方差方法
//    double spectrum_score = calcSharpnessSpectrumScore(image);  // 频谱方法
//    double energy_score = calcSharpnessEnergyScore(image);      // 能量方法

    double old_score = calcSharpnessOld(&image);

    // 输出清晰度值
    cout << "Gradient Score: " << gradient_score << endl;
//    cout << "Variance Score: " << variance_score << endl;
//    cout << "Spectrum Score: " << spectrum_score << endl;
//    cout << "Energy Score: " << energy_score << endl;

    cout << "Gradient Score: " << old_score << endl;

    return gradient_score;
}
