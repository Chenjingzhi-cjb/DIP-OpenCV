#include "other_application.h"


// Phase Correlation
cv::Point2d
matchTemplateSubPixelByPhaseCorrelation(const cv::Mat &image_tmpl, const cv::Mat &image_dst, const cv::Point &max_loc) {
    if (max_loc.x + image_tmpl.cols > image_dst.cols || max_loc.y + image_tmpl.rows > image_dst.rows) {
        std::cout << "calcTemplatePosition() Error: The max_roi out of bounds for phase correlation" << std::endl;
        return {0.0, 0.0};
    }

    cv::Mat tmpl_32FC1, dst_32FC1;
    image_tmpl.convertTo(tmpl_32FC1, CV_32F);
    image_dst.convertTo(dst_32FC1, CV_32F);
    if (tmpl_32FC1.channels() > 1) {
        cv::cvtColor(tmpl_32FC1, tmpl_32FC1, cv::COLOR_BGR2GRAY);
    }
    if (dst_32FC1.channels() > 1) {
        cv::cvtColor(dst_32FC1, dst_32FC1, cv::COLOR_BGR2GRAY);
    }

    // 使用基于“FFT相位相关”和“加权质心亚像素插值”的算法来进行精确定位
    cv::Rect max_roi(max_loc.x, max_loc.y, image_tmpl.cols, image_tmpl.rows);
    cv::Point2d subpixel_offset = cv::phaseCorrelate(dst_32FC1(max_roi), tmpl_32FC1);

    return subpixel_offset;
}

// Corner - Grayscale Centroid Iterative Fitting
cv::Point2d matchTemplateSubPixelByCorner(const cv::Mat &result, const cv::Point &max_loc) {
    int criteria_maxCount = 40;
    double criteria_epsilon = 0.01;

    cv::Size winSize(5, 5);
    cv::Size zeroZone(-1, -1);

    if (max_loc.x > 2 && max_loc.x < result.cols - 2 && max_loc.y > 2 && max_loc.y < result.rows - 2) {
        // 归一化
        cv::Mat result_norm;
        cv::normalize(result, result_norm, 0, 255, cv::NORM_MINMAX, CV_32F);

        // 设置初始点
        std::vector<cv::Point2f> corners{cv::Point2f((float) max_loc.x, (float) max_loc.y)};

        // 设置迭代停止条件
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, criteria_maxCount,
                                  criteria_epsilon);

        // 进行迭代优化
        cv::cornerSubPix(result_norm, corners, winSize, zeroZone, criteria);

        // 统一返回增量
        return {corners[0].x - (float) max_loc.x, corners[0].y - (float) max_loc.y};
    } else {
        std::cout << "calcTemplatePosition() Error: Peak too close to edge, skipping sub-pixel refinement."
                  << std::endl;
    }

    return {0.0, 0.0};
}

// Quadratic Curve Interpolation Fitting
cv::Point2d matchTemplateSubPixelByQuadInterp(const cv::Mat &result, const cv::Point &max_loc) {
    // 确保 max_loc 不在边界
    if (max_loc.x > 0 && max_loc.x < result.cols - 1 && max_loc.y > 0 && max_loc.y < result.rows - 1) {
        // 提取 中心点 和 X、Y 方向邻域 的 响应值
        float center = result.at<float>(max_loc.y, max_loc.x);
        float left = result.at<float>(max_loc.y, max_loc.x - 1);
        float right = result.at<float>(max_loc.y, max_loc.x + 1);
        float up = result.at<float>(max_loc.y - 1, max_loc.x);
        float down = result.at<float>(max_loc.y + 1, max_loc.x);

        // 计算 X 方向的亚像素偏移，注意如果分母接近0，说明曲线是平的或线性的，无法计算偏移
        double dx = 0.0;
        double denominator_x = 2 * (left - 2 * center + right);
        if (std::abs(denominator_x) > 1e-6) {
            dx = (left - right) / denominator_x;
        }

        // 计算 Y 方向的亚像素偏移，注意如果分母接近0，说明曲线是平的或线性的，无法计算偏移
        double dy = 0.0;
        double denominator_y = 2 * (up - 2 * center + down);
        if (std::abs(denominator_y) > 1e-6) {
            dy = (up - down) / denominator_y;
        }

        // 计算结果检查，偏移量应该在 [-0.5, 0.5] 范围内，如果超出，说明该峰值可能不是真正的极值点
        if (std::abs(dx) <= 0.5 && std::abs(dy) <= 0.5) {
            return {dx, dy};
        }
    } else {
        std::cout << "calcTemplatePosition() Error: Peak is on the edge, cannot perform sub-pixel refinement."
                  << std::endl;
    }

    return {0.0, 0.0};
}

// 2D Gaussian Fitting
cv::Point2d matchTemplateSubPixelByGaussian(const cv::Mat &result, const cv::Point &max_loc) {
    // 确保 max_loc 不在边界
    if (max_loc.x > 0 && max_loc.x < result.cols - 1 && max_loc.y > 0 && max_loc.y < result.rows - 1) {
        // 提取 3x3 的邻域
        cv::Mat patch = result(cv::Rect(max_loc.x - 1, max_loc.y - 1, 3, 3)).clone();

        // 对数变换前先检查所有值是否为正，如果有负值或零值则进行偏移使所有值为正
        double min_patch_val;
        cv::minMaxLoc(patch, &min_patch_val, nullptr, nullptr, nullptr);
        if (min_patch_val <= 0) {
            patch += (-min_patch_val + 1e-3);
        }

        // 对数变换
        cv::Mat log_patch;
        cv::log(patch, log_patch);

        // 使用中心差分近似求二阶偏导
        double dx = 0.5 * (log_patch.at<float>(1, 2) - log_patch.at<float>(1, 0));
        double dy = 0.5 * (log_patch.at<float>(2, 1) - log_patch.at<float>(0, 1));
        double dxx = log_patch.at<float>(1, 0) - 2 * log_patch.at<float>(1, 1) + log_patch.at<float>(1, 2);
        double dyy = log_patch.at<float>(0, 1) - 2 * log_patch.at<float>(1, 1) + log_patch.at<float>(2, 1);
        double dxy = 0.25 * (log_patch.at<float>(2, 2) - log_patch.at<float>(2, 0) - log_patch.at<float>(0, 2) +
                             log_patch.at<float>(0, 0));

        // 检查 Hessian 矩阵是否为负定（确保是极大值点）
        double det = dxx * dyy - dxy * dxy;
        if (det > 1e-6 && dxx < -1e-6 && dyy < -1e-6) {  // 负定矩阵条件
            try {
                // 解二次曲面极值位置: H * delta = -grad
                cv::Matx22d H(dxx, dxy, dxy, dyy);
                cv::Vec2d g(dx, dy);
                cv::Vec2d delta = -H.solve(g, cv::DECOMP_SVD);

                // 限制偏移量在合理范围内
                if (std::abs(delta[0]) <= 0.5 && std::abs(delta[1]) <= 0.5) {
                    return {delta[0], delta[1]};
                }
            } catch (const cv::Exception &e) {
                std::cout << "calcTemplatePosition() Error: Matrix solve failed: " << e.what() << std::endl;
            }
        }
    } else {
        std::cout << "calcTemplatePosition() Error: Peak is on the edge, cannot perform sub-pixel refinement."
                  << std::endl;
    }

    return {0.0, 0.0};
}

double
calcTemplatePosition(const cv::Mat &image_tmpl, const cv::Mat &image_dst, cv::Point2d &position, bool sub_pixel) {
    if (image_tmpl.empty() || image_dst.empty()) {
        throw std::invalid_argument("calcTemplatePosition() Error: Input image is empty!");
    }

    if (image_tmpl.cols > image_dst.cols || image_tmpl.rows > image_dst.rows) {
        throw std::invalid_argument(
                "calcTemplatePosition() Error: Template dimensions must be smaller than or equal to the destination image.");
    }

    // 进行模板匹配
    cv::Mat result;
    cv::matchTemplate(image_dst, image_tmpl, result, cv::TM_CCOEFF_NORMED);

    // 定位匹配的位置
    double max_val;  // 匹配得分，用于判断是否成功匹配
    cv::Point max_loc;
    cv::minMaxLoc(result, nullptr, &max_val, nullptr, &max_loc);

    double position_x = max_loc.x;
    double position_y = max_loc.y;

    // 亚像素级
    if (sub_pixel) {
        cv::Point2d subpixel_offset = matchTemplateSubPixelByQuadInterp(result, max_loc);
//        cv::Point2d subpixel_offset = matchTemplateSubPixelByGaussian(result, max_loc);

        position_x += subpixel_offset.x;
        position_y += subpixel_offset.y;
    }

    // 计算结果，中心坐标系像素
    position.x = position_x - (image_dst.cols - image_tmpl.cols) / 2.0;
    position.y = (image_dst.rows - image_tmpl.rows) / 2.0 - position_y;

    return max_val;
}

double calcImageOffset(const cv::Mat &image_src, const cv::Mat &image_dst, cv::Point2d &offset) {
    if (image_src.empty() || image_dst.empty()) {
        throw std::invalid_argument("calcImageOffset() Error: Input image is empty!");
    }

    if (image_src.size() != image_dst.size()) {
        throw std::invalid_argument("calcImageOffset() Error: Input images must have the same dimensions.");
    }

    // 使用基于“FFT相位相关”和“加权质心亚像素插值”的算法来计算偏移量
    double response;
    cv::Point2f subpixel_offset = cv::phaseCorrelate(image_src, image_dst, cv::noArray(), &response);

    // 返回结果，中心坐标系像素
    offset.x = subpixel_offset.x;
    offset.y = -subpixel_offset.y;

    return response;
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

double calcImageSharpness(Mat &image) {
    if (image.empty()) {
        throw invalid_argument("calcImageSharpness(): Input image is empty!");
    }

    double gradient_score = calcSharpnessGradientScore(image);  // 梯度方法
//    double variance_score = calcSharpnessVarianceScore(image);  // 方差方法
//    double spectrum_score = calcSharpnessSpectrumScore(image);  // 频谱方法
//    double energy_score = calcSharpnessEnergyScore(image);      // 能量方法

    // 输出清晰度值
    cout << "Gradient Score: " << gradient_score << endl;
//    cout << "Variance Score: " << variance_score << endl;
//    cout << "Spectrum Score: " << spectrum_score << endl;
//    cout << "Energy Score: " << energy_score << endl;

    return gradient_score;
}

double calcSharpnessOldOpt(cv::Mat *image, int part_count) {
    if (part_count % 2 != 0) return 0;

    std::vector<cv::Point> points;

    // get points
    for (int i = 3; i < image->rows - 3; i++) {
        int row_max_pixel = 0;
        int row_min_pixel = 255;

        auto *row = image->ptr<uchar>(i);

        /* old
        for (int j = 3; j < image->cols - 3; j++) {
            bool is_max = false, is_min = false;

            if (row[j] > row_max_pixel) {
                row_max_pixel = row[j];
                is_max = true;
            }
            if (row[j] < row_min_pixel) {
                row_min_pixel = row[j];
                is_min = true;
            }

            if (is_max || is_min) {
                points.emplace_back(cv::Point(i, j));
//                image->at<uchar>(i, j) = 80;  // see points
            }
        }
        */

        // get points opt++
        int start_j, end_j;

        // left
        for (int part_num = part_count / 2; part_num > 0; part_num--) {
            row_max_pixel = 0;
            row_min_pixel = 255;

            start_j = image->cols * part_num / part_count;
            end_j = image->cols * (part_num - 1) / part_count;
            if (part_num == 1) end_j = 2;

            for (int j = start_j; j > end_j; j--) {
                bool is_max = false, is_min = false;

                if (row[j] > row_max_pixel) {
                    row_max_pixel = row[j];
                    is_max = true;
                }
                if (row[j] < row_min_pixel) {
                    row_min_pixel = row[j];
                    is_min = true;
                }

                if (is_max || is_min) {
                    points.emplace_back(cv::Point(i, j));
//                    image->at<uchar>(i, j) = 80;  // see points
                }
            }
        }

        // right
        for (int part_num = part_count / 2; part_num < part_count; part_num++) {
            row_max_pixel = 0;
            row_min_pixel = 255;

            start_j = image->cols * part_num / part_count;
            end_j = image->cols * (part_num + 1) / part_count;
            if (part_num == part_count - 1) end_j = image->cols - 3;

            for (int j = start_j; j < end_j; j++) {
                bool is_max = false, is_min = false;

                if (row[j] > row_max_pixel) {
                    row_max_pixel = row[j];
                    is_max = true;
                }
                if (row[j] < row_min_pixel) {
                    row_min_pixel = row[j];
                    is_min = true;
                }

                if (is_max || is_min) {
                    points.emplace_back(cv::Point(i, j));
//                    image->at<uchar>(i, j) = 80;  // see points
                }
            }
        }
    }

    // see points
//    cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
//    cv::imshow("image", *image);
//    cv::waitKey(0);

    // calculate value
    double S_vertical = 0, S_horizontal = 0;
    for (auto &i: points) {
        double vertical_gradient, horizontal_gradient;
        int col_no = i.y;
        int row_no = i.x;

        auto *plus_last_row = image->ptr<uchar>(row_no - 2);
        auto *last_row = image->ptr<uchar>(row_no - 1);
        auto *current_row = image->ptr<uchar>(row_no);
        auto *next_row = image->ptr<uchar>(row_no + 1);
        auto *plus_next_row = image->ptr<uchar>(row_no + 2);

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

//    std::cout << (int) points.size() << std::endl;
//    std::cout << S << std::endl;
//    getAndPrintTime();

    return S;
}
