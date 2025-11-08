#include "image_segmentation.h"


void pointDetectLaplaceKernel(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    // 孤立点检测的拉普拉斯核
    cv::Mat kernel = (cv::Mat_<char>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);

    cv::filter2D(src, dst, src.depth(), kernel);
}

void lineDetectLaplaceKernel(const cv::Mat &src, cv::Mat &dst, int line_type) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    // 线检测的拉普拉斯核
    cv::Mat kernel;
    if (line_type == 1) {
        kernel = (cv::Mat_<char>(3, 3) << -1, -1, -1, 2, 2, 2, -1, -1, -1);  // 水平
    } else if (line_type == 2) {
        kernel = (cv::Mat_<char>(3, 3) << 2, -1, -1, -1, 2, -1, -1, -1, 2);  // 西北-东南 方向
    } else if (line_type == 3) {
        kernel = (cv::Mat_<char>(3, 3) << -1, 2, -1, -1, 2, -1, -1, 2, -1);  // 垂直
    } else if (line_type == 4) {
        kernel = (cv::Mat_<char>(3, 3) << -1, -1, 2, -1, 2, -1, 2, -1, -1);  // 东北-西南 方向
    } else {
        THROW_ARG_ERROR("Invalid `line_type`: {}.", line_type);
    }

    cv::filter2D(src, dst, src.depth(), kernel);
}

void lineDetectHough(const cv::Mat &src, cv::Mat &dst, double rho, double theta, int threshold, double srn,
                     double stn, double min_theta, double max_theta) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(src, lines, rho, theta, threshold, srn, stn, min_theta, max_theta);

    cv::Mat temp = src.clone();
    for (const auto &l: lines) {
        float line_rho = l[0], line_theta = l[1];
        double a = cos(line_theta), b = sin(line_theta);
        double x0 = a * line_rho, y0 = b * line_rho;
        cv::Point p1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        cv::Point p2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
        cv::line(temp, p1, p2, cv::Scalar(255, 0, 0), 3);
    }

    temp.copyTo(dst);
}

void lineSegmentDetectHough(const cv::Mat &src, cv::Mat &dst, double rho, double theta, int threshold,
                            double minLineLength, double maxLineGap) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(src, lines, rho, theta, threshold, minLineLength, maxLineGap);

    cv::Mat temp = src.clone();
    for (const auto &l: lines) {
        cv::Point p1(l[0], l[1]);
        cv::Point p2(l[2], l[3]);
        cv::line(temp, p1, p2, cv::Scalar(255, 0, 0), 3);
    }

    temp.copyTo(dst);
}

void circleDetectHough(const cv::Mat &src, cv::Mat &dst, int method, double dp, double minDist, double param1,
                       double param2, int minRadius, int maxRadius) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(src, circles, method, dp, minDist, param1, param2, minRadius, maxRadius);

    cv::Mat temp = src.clone();
    for (const auto &c: circles) {
        cv::Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);

        // 绘制圆心
        cv::circle(temp, center, 3, cv::Scalar(0, 255, 0), cv::FILLED);
        // 绘制圆轮廓
        cv::circle(temp, center, radius, cv::Scalar(255, 0, 0), 3);
    }

    temp.copyTo(dst);
}

void cornerDetectHarris(const cv::Mat &src, cv::Mat &dst, int threshold, int blockSize, int ksize, double k,
                        int borderType) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8U && src.type() != CV_32F) {
        THROW_ARG_ERROR("Input src image type must be CV_8U or CV_32F.");
    }

    cv::Mat temp;
    if (src.channels() == 1) {
        temp = src.clone();
    } else if (src.channels() == 3) {
        bgrToGray(src, temp);
    } else {
        THROW_ARG_ERROR("The number of channels for the input src image is not supported: {}", src.channels());
    }

    // 计算 Harris 值 并 归一化
    cv::Mat harris_value;
    cv::cornerHarris(temp, harris_value, blockSize, ksize, k, borderType);
    cv::normalize(harris_value, harris_value, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // 阈值处理 并 绘制角点
    temp = src.clone();
    if (temp.channels() == 1) {
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.cols; j++) {
                if (harris_value.at<uchar>(i, j) > threshold) {
                    cv::circle(temp, cv::Point(j, i), 2, cv::Scalar(255), 2, 8, 0);
                }
            }
        }
    } else if (temp.channels() == 3) {
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.cols; j++) {
                if (harris_value.at<uchar>(i, j) > threshold) {
                    cv::circle(temp, cv::Point(j, i), 2, cv::Scalar(0, 255, 0), 2, 8, 0);
                }
            }
        }
    }

    temp.copyTo(dst);
}

void cornerDetectShiTomasi(const cv::Mat &src, cv::Mat &dst, int maxCorners, double qualityLevel, double minDistance,
                           cv::InputArray mask, int blockSize) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8U && src.type() != CV_32F) {
        THROW_ARG_ERROR("Input src image type must be CV_8U or CV_32F.");
    }

    cv::Mat temp;
    if (src.channels() == 1) {
        temp = src.clone();
    } else if (src.channels() == 3) {
        bgrToGray(src, temp);
    } else {
        THROW_ARG_ERROR("The number of channels for the input src image is not supported: {}", src.channels());
    }

    // 计算角点坐标
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(temp, corners, maxCorners, qualityLevel, minDistance, mask, blockSize);

    // 绘制角点
    temp = src.clone();
    if (temp.channels() == 1) {
        for (const auto &corner: corners) {
            cv::circle(temp, corner, 2, cv::Scalar(255), 2, 8, 0);
        }
    } else if (temp.channels() == 3) {
        for (const auto &corner: corners) {
            cv::circle(temp, corner, 2, cv::Scalar(0, 255, 0), 2, 8, 0);
        }
    }

    temp.copyTo(dst);
}

void cornerDetectSubPixel(const cv::Mat &src, cv::Mat &dst, int maxCorners, double qualityLevel, double minDistance,
                          cv::Size winSize, cv::Size zeroZone, cv::TermCriteria criteria, cv::InputArray mask,
                          int blockSize, bool useHarrisDetector, double k) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8U && src.type() != CV_32F) {
        THROW_ARG_ERROR("Input src image type must be CV_8U or CV_32F.");
    }

    cv::Mat temp;
    if (src.channels() == 1) {
        temp = src.clone();
    } else if (src.channels() == 3) {
        bgrToGray(src, temp);
    } else {
        THROW_ARG_ERROR("The number of channels for the input src image is not supported: {}", src.channels());
    }

    // 计算角点坐标
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(temp, corners, maxCorners, qualityLevel, minDistance, mask, blockSize,
                            useHarrisDetector, k);
    // 计算亚像素级角点坐标
    cv::cornerSubPix(temp, corners, winSize, zeroZone, criteria);

    // 绘制角点
    temp = src.clone();
    if (temp.channels() == 1) {
        for (const auto &corner: corners) {
            cv::circle(temp, corner, 2, cv::Scalar(255), 2, 8, 0);
        }
    } else if (temp.channels() == 3) {
        for (const auto &corner: corners) {
            cv::circle(temp, corner, 2, cv::Scalar(0, 255, 0), 2, 8, 0);
        }
    }

    temp.copyTo(dst);
}

int calcGlobalThresholdClassMean(const cv::Mat &src, const cv::Mat &mask) {
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

    // 前缀和
    std::vector<float> num(hist_size), value(hist_size);
    num[0] = hist.at<float>(0);
    value[0] = 0.0f;
    for (int i = 1; i < hist_size; i++) {
        num[i] = num[i - 1] + hist.at<float>(i);
        value[i] = value[i - 1] + hist.at<float>(i) * static_cast<float>(i);
    }

    float total_num = num.back();
    float total_value = value.back();

    // 遍历阈值，找最优
    int best_thresh = 0;
    float best_diff = std::numeric_limits<float>::max();

    for (int i = 1; i < hist_size - 1; i++) {
        float num_fg = num[i];
        float num_bg = total_num - num_fg;
        if (num_fg == 0 || num_bg == 0) continue;

        float mean_fg = value[i] / num_fg;
        float mean_bg = (total_value - value[i]) / num_bg;
        float diff = std::abs(static_cast<float>(i) - 0.5f * (mean_fg + mean_bg));

        if (diff < best_diff) {
            best_diff = diff;
            best_thresh = i;
        }
    }

    return best_thresh;
}

int calcGlobalThresholdOtus(const cv::Mat &src, const cv::Mat &mask, double *eta) {
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

    int total = mask.empty() ? (int) src.total() : cv::countNonZero(mask);

    // 转换为概率分布
    std::vector<double> prob(hist_size);
    for (int i = 0; i < hist_size; ++i)
        prob[i] = hist.at<float>(i) / (float) total;

    // 累积概率与均值
    std::vector<double> omega(hist_size, 0.0);  // 累积权重
    std::vector<double> mu(hist_size, 0.0);     // 累积均值

    omega[0] = prob[0];
    mu[0] = 0.0;
    for (int i = 1; i < hist_size; ++i) {
        omega[i] = omega[i - 1] + prob[i];
        mu[i] = mu[i - 1] + i * prob[i];
    }

    double mu_total = mu[hist_size - 1];

    // 计算全局方差
    double sigma_total = 0.0;
    for (int i = 0; i < hist_size; ++i) {
        sigma_total += (i - mu_total) * (i - mu_total) * prob[i];
    }

    // 遍历所有阈值计算类间方差
    double max_sigma = -1.0;
    std::vector<int> best_thresh_list;
    for (int t = 0; t < (hist_size - 1); ++t) {
        if (omega[t] <= 1e-6 || omega[t] >= (1.0 - 1e-6))
            continue;

        double N = std::pow(mu_total * omega[t] - mu[t], 2);
        double D = omega[t] * (1 - omega[t]);
        double sigma = N / D;

        if (std::abs(sigma - max_sigma) < 1e-12) {
            // 有相同的极大值，记录
            best_thresh_list.push_back(t);
        } else if (sigma > max_sigma) {
            // 有更大的极大值，更新
            max_sigma = sigma;
            best_thresh_list.clear();
            best_thresh_list.push_back(t);
        }
    }

    // 若极大值不唯一，取平均
    double avg_thresh = 0.0;
    for (int t: best_thresh_list)
        avg_thresh += t;
    avg_thresh /= (double) best_thresh_list.size();

    // 可分离性测度
    if (eta && sigma_total > 1e-12)
        *eta = max_sigma / sigma_total;
    else if (eta)
        *eta = 0.0;

    return (int) std::round(avg_thresh);
}

int getPercentileGrayValue(const cv::Mat &src, double percentile) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    // 展平为一维数组
    std::vector<uchar> pixels;
    pixels.assign(src.datastart, src.dataend);

    // 排序
    std::sort(pixels.begin(), pixels.end());

    // 计算索引
    auto index = static_cast<size_t>(percentile * (double) (pixels.size() - 1));

    return pixels[index];
}

int calcGlobalThresholdEdgeOpt(const cv::Mat &src, int gradient_mode, double percentile, int threshold_mode) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    // 使用 Sobel or Laplacian 计算梯度
    cv::Mat gradient;
    if (gradient_mode == 1) {
        cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
        cv::Sobel(src, grad_x, CV_16S, 1, 0, 3);
        cv::convertScaleAbs(grad_x, abs_grad_x);
        cv::Sobel(src, grad_y, CV_16S, 0, 1, 3);
        cv::convertScaleAbs(grad_y, abs_grad_y);
        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradient);
    } else if (gradient_mode == 2) {
        // Laplacian
        cv::Mat laplacian_dst;
        cv::Laplacian(src, laplacian_dst, CV_16S, 3);
        cv::convertScaleAbs(laplacian_dst, gradient);
    } else {
        THROW_ARG_ERROR("Invalid `gradient_mode`: {}.", gradient_mode);
    }
    cv::normalize(gradient, gradient, 0, 255, cv::NORM_MINMAX);

    // 观察梯度图像
//    cv::namedWindow("gradient", cv::WINDOW_AUTOSIZE);
//    cv::imshow("gradient", gradient);
//    cv::waitKey(0);

    // 根据百分位数对梯度图像进行二值化，作掩模
    int thresh = getPercentileGrayValue(gradient, percentile);
    cv::threshold(gradient, gradient, thresh, 255, cv::THRESH_BINARY);

    // 观察梯度二值图像
//    cv::namedWindow("gradient", cv::WINDOW_AUTOSIZE);
//    cv::imshow("gradient", gradient);
//    cv::waitKey(0);

    // 观察掩模后的直方图
//    cv::Mat hist_mask = grayHistogram(src, gradient);
//    cv::namedWindow("hist_mask", cv::WINDOW_AUTOSIZE);
//    cv::imshow("hist_mask", hist_mask);
//    cv::waitKey(0);

    // 计算掩模后的阈值
    int t;
    if (threshold_mode == 1) {
        t = calcGlobalThresholdClassMean(src, gradient);
    } else if (threshold_mode == 2) {
        t = calcGlobalThresholdOtus(src, gradient, nullptr);
    } else {
        THROW_ARG_ERROR("Invalid `threshold_mode`: {}.", threshold_mode);
    }

    return t;
}

std::pair<int, int> calcGlobalDualThresholdOtus(const cv::Mat &src, const cv::Mat &mask, double *eta) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    // 计算灰度直方图
    int hist_size = 256;
    float range[] = {0, 256};
    const float *hist_range = {range};

    cv::Mat hist;
    cv::calcHist(&src, 1, nullptr, mask, hist, 1, &hist_size, &hist_range);

    int total = mask.empty() ? (int) src.total() : cv::countNonZero(mask);

    // 转为概率分布
    std::vector<double> prob(hist_size);
    for (int i = 0; i < hist_size; ++i)
        prob[i] = hist.at<float>(i) / (double) total;

    // 累积概率与累积均值
    std::vector<double> omega(hist_size, 0.0);
    std::vector<double> mu(hist_size, 0.0);

    omega[0] = prob[0];
    mu[0] = 0.0;
    for (int i = 1; i < hist_size; ++i) {
        omega[i] = omega[i - 1] + prob[i];
        mu[i] = mu[i - 1] + i * prob[i];
    }

    double mu_total = mu[hist_size - 1];

    // 计算全局方差
    double sigma_total = 0.0;
    for (int i = 0; i < hist_size; ++i)
        sigma_total += (i - mu_total) * (i - mu_total) * prob[i];

    // 双阈值遍历，t1 < t2，计算类间方差
    double max_sigma = -1.0;
    std::vector<std::pair<int, int>> best_thresh_list;
    for (int t1 = 0; t1 < hist_size - 2; ++t1) {
        for (int t2 = t1 + 1; t2 < hist_size - 1; ++t2) {
            double w0 = omega[t1];
            double w1 = omega[t2] - omega[t1];
            double w2 = 1.0 - omega[t2];

            if (w0 <= 1e-6 || w1 <= 1e-6 || w2 <= 1e-6)
                continue;

            double m0 = mu[t1] / w0;
            double m1 = (mu[t2] - mu[t1]) / w1;
            double m2 = (mu[hist_size - 1] - mu[t2]) / w2;

            double between = w0 * std::pow(m0 - mu_total, 2) +
                             w1 * std::pow(m1 - mu_total, 2) +
                             w2 * std::pow(m2 - mu_total, 2);

            if (std::abs(between - max_sigma) < 1e-12) {
                best_thresh_list.emplace_back(t1, t2);
            } else if (between > max_sigma) {
                max_sigma = between;
                best_thresh_list.clear();
                best_thresh_list.emplace_back(t1, t2);
            }
        }
    }

    // 若存在多个极值，取平均
    double avg_t1 = 0.0, avg_t2 = 0.0;
    for (auto &p: best_thresh_list) {
        avg_t1 += p.first;
        avg_t2 += p.second;
    }
    avg_t1 /= (double) best_thresh_list.size();
    avg_t2 /= (double) best_thresh_list.size();

    // 可分离性测度
    if (eta && sigma_total > 1e-12)
        *eta = max_sigma / sigma_total;
    else if (eta)
        *eta = 0.0;

    return {(int) std::round(avg_t1), (int) std::round(avg_t2)};
}

void thresholdThreeClass(const cv::Mat &src, cv::Mat &dst, int t1, int t2) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    int low, high;
    if (t1 < t2) {
        low = t1;
        high = t2;
    } else {
        low = t2;
        high = t1;
    }

    cv::Mat temp = cv::Mat::zeros(src.size(), CV_8UC1);
    temp.setTo(0, src <= low);
    temp.setTo(127, (src > low) & (src <= high));
    temp.setTo(255, src > high);

    temp.copyTo(dst);
}
