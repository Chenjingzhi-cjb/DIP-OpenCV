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

    return ((max_val + 1.0) / 2.0);  // [0, 1]
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

double calcTemplatePositionByMatches(const cv::Mat &image_tmpl, const cv::Mat &image_dst, cv::Point2d &position,
                                     double *scale, double *angle, std::vector<cv::KeyPoint> &key_points_tmpl,
                                     std::vector<cv::KeyPoint> &key_points_dst, std::vector<cv::DMatch> &good_matches,
                                     bool show_matches = false) {
    if (good_matches.size() < 4) {
        std::cout << "calcTemplatePositionByMatches() Error: Not enough good matches to estimate transform."
                  << std::endl;
        return 0;
    }

    // 提取优质匹配点的位置
    std::vector<cv::Point2f> good_points_tmpl, good_points_dst;
    for (const auto &good_match : good_matches) {
        good_points_tmpl.push_back(key_points_tmpl[good_match.queryIdx].pt);
        good_points_dst.push_back(key_points_dst[good_match.trainIdx].pt);
    }

    // 匹配可视化
    if (show_matches) {
        cv::Mat image_matches;
        cv::drawMatches(image_tmpl, key_points_tmpl, image_dst, key_points_dst, good_matches, image_matches,
                        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("Good Matches", image_matches);
        cv::waitKey(0);
    }

    // 使用 RANSAC 算法估计仿射变换矩阵 M
    // dst_pt = M * tmpl_pt
    std::vector<uchar> inliers_mask;
    cv::Mat M = cv::estimateAffinePartial2D(good_points_tmpl, good_points_dst, inliers_mask);

    if (M.empty()) {
        std::cout << "calcTemplatePositionByMatches() Error: Transform estimation failed." << std::endl;
        return 0;
    }

    // 计算置信度：内点数 / 良好匹配点总数
    int inliers_count = cv::countNonZero(inliers_mask);
    double confidence = static_cast<double>(inliers_count) / (double) good_matches.size();

    // 解析变换矩阵参数
    // M = [[s*cos(a), -s*sin(a), tx], [s*sin(a), s*cos(a), ty]]
    // a = atan2(sin(a), cos(a))
    double m00 = M.at<double>(0, 0);
    double m01 = M.at<double>(0, 1);
    double m10 = M.at<double>(1, 0);
    if (scale) *scale = std::sqrt(m00 * m00 + m01 * m01);
    if (angle) *angle = std::atan2(m10, m00) * 180.0 / CV_PI;

    // 使用变换矩阵 M 进行坐标变换
    std::vector<cv::Point2f> position_tmpl_center;
    cv::transform(std::vector<cv::Point2f>{{(float) image_tmpl.cols / 2.0f, (float) image_tmpl.rows / 2.0f}},
                  position_tmpl_center, M);

    // 计算结果，中心坐标系像素
    position.x = position_tmpl_center[0].x - image_dst.cols / 2.0;
    position.y = image_dst.rows / 2.0 - position_tmpl_center[0].y;

    return confidence;
}

void cornerDetectAndCompute(const cv::Mat &image, std::vector<cv::KeyPoint> &key_points, cv::Mat &descriptors,
                            CornerDescriptorType descriptor_type, int maxCorners, double qualityLevel,
                            double minDistance, cv::Size winSize, cv::TermCriteria criteria, int min_Corners,
                            float keypoint_diameter) {
    // 检测角点
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(image, corners, maxCorners, qualityLevel, minDistance);
    if (corners.size() < min_Corners) {
        std::cout << "cornerDetectAndCompute() Error: Insufficient number of corner points in image!" << std::endl;
        return;
    }

    // 角点亚像素精度优化
    cv::cornerSubPix(image, corners, winSize, cv::Size(-1, -1), criteria);

    // 计算特征点
    std::vector<cv::KeyPoint> _key_points;
    for (const auto &corner : corners) {
        _key_points.emplace_back(corner, keypoint_diameter);
    }

    // 计算描述子
    cv::Mat _descriptors;
    if (descriptor_type == CornerDescriptorType::ORB) {
        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        orb->compute(image, _key_points, _descriptors);
    } else if (descriptor_type == CornerDescriptorType::SIFT) {
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        sift->compute(image, _key_points, _descriptors);
    }

    // 返回结果
    key_points = _key_points;
    descriptors = _descriptors;
}

bool
calcGoodMatchesByBFMatcher(std::vector<cv::KeyPoint> &key_points_tmpl, std::vector<cv::KeyPoint> &key_points_dst,
                           cv::Mat &descriptors_tmpl, cv::Mat &descriptors_dst,
                           std::vector<cv::DMatch> &good_matches) {
    if (key_points_tmpl.empty() || key_points_dst.empty() || descriptors_tmpl.empty() || descriptors_dst.empty()) {
        std::cout << "calcGoodMatchesByBFMatcher() Error: Not enough key points detected in one of the images."
                  << std::endl;
        return false;
    }

    // 使用 Brute-Force Matcher 进行特征匹配，计算汉明距离
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors_tmpl, descriptors_dst, matches);

    if (matches.empty()) {
        std::cout << "calcGoodMatchesByBFMatcher() Error: No matches found." << std::endl;
        return false;
    }

    // 根据 匹配距离 [min_dis, 2 * min_dis] 筛选 优质匹配点
    good_matches.clear();
    std::sort(matches.begin(), matches.end(),
              [](const cv::DMatch &a, const cv::DMatch &b) { return a.distance < b.distance; });
    for (auto &match : matches) {
        if (match.distance >= (2 * matches[0].distance + 1e-3)) break;
        good_matches.push_back(match);
    }

    return true;
}

bool
calcGoodMatchesByFLANNMatcher(std::vector<cv::KeyPoint> &key_points_tmpl, std::vector<cv::KeyPoint> &key_points_dst,
                              cv::Mat &descriptors_tmpl, cv::Mat &descriptors_dst,
                              std::vector<cv::DMatch> &good_matches) {
    if (key_points_tmpl.empty() || key_points_dst.empty() || descriptors_tmpl.empty() || descriptors_dst.empty()) {
        std::cout << "calcGoodMatchesByFLANNMatcher() Error: Not enough key points detected in one of the images."
                  << std::endl;
        return false;
    }

    // 使用 FLANN Matcher 进行特征匹配
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::FlannBasedMatcher::create();
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descriptors_tmpl, descriptors_dst, knn_matches, 2);

    if (knn_matches.empty()) {
        std::cout << "calcGoodMatchesByFLANNMatcher() Error: No matches found." << std::endl;
        return false;
    }

    // 使用 Lowe's ratio test 匹配策略
    good_matches.clear();
    const float ratio_thresh = 0.7f;
    for (auto &match : knn_matches) {
        if (match.size() >= 2 &&
            match[0].distance < ratio_thresh * match[1].distance) {
            good_matches.push_back(match[0]);
        }
    }

    return true;
}

double
calcTemplatePositionORB(const cv::Mat &image_tmpl, const cv::Mat &image_dst, cv::Point2d &position, double *scale,
                        double *angle, int pre_features_num) {
    if (image_tmpl.empty() || image_dst.empty()) {
        throw std::invalid_argument("calcTemplatePositionORB() Error: Input image is empty!");
    }

    if (image_tmpl.cols > image_dst.cols || image_tmpl.rows > image_dst.rows) {
        throw std::invalid_argument(
                "calcTemplatePositionORB() Error: Template dimensions must be smaller than or equal to the destination image.");
    }

    // 初始化 ORB 检测器
    cv::Ptr<cv::ORB> orb = cv::ORB::create(pre_features_num);

    // 检测并计算 模板图像 和 目标图像 的特征点与描述子
    std::vector<cv::KeyPoint> key_points_tmpl, key_points_dst;
    cv::Mat descriptors_tmpl, descriptors_dst;
    orb->detectAndCompute(image_tmpl, cv::noArray(), key_points_tmpl, descriptors_tmpl);
    orb->detectAndCompute(image_dst, cv::noArray(), key_points_dst, descriptors_dst);

    std::vector<cv::DMatch> good_matches;
    if (!calcGoodMatchesByBFMatcher(key_points_tmpl, key_points_dst, descriptors_tmpl, descriptors_dst,
                                    good_matches))
        return 0;

    return calcTemplatePositionByMatches(image_tmpl, image_dst, position, scale, angle, key_points_tmpl,
                                         key_points_dst, good_matches);
}

double
calcTemplatePositionCorner(const cv::Mat &image_tmpl, const cv::Mat &image_dst, cv::Point2d &position, double *scale,
                           double *angle, CornerDescriptorType descriptor_type) {
    if (image_tmpl.empty() || image_dst.empty()) {
        throw std::invalid_argument("calcTemplatePositionCorner() Error: Input image is empty!");
    }

    if (image_tmpl.cols > image_dst.cols || image_tmpl.rows > image_dst.rows) {
        throw std::invalid_argument(
                "calcTemplatePositionCorner() Error: Template dimensions must be smaller than or equal to the destination image.");
    }

    // 检测模板图像和目标图像的角点并计算特征点与描述子
    std::vector<cv::KeyPoint> key_points_tmpl, key_points_dst;
    cv::Mat descriptors_tmpl, descriptors_dst;
    cornerDetectAndCompute(image_tmpl, key_points_tmpl, descriptors_tmpl, descriptor_type);
    cornerDetectAndCompute(image_dst, key_points_dst, descriptors_dst, descriptor_type);

    std::vector<cv::DMatch> good_matches;
    if (descriptor_type == CornerDescriptorType::ORB) {
        if (!calcGoodMatchesByBFMatcher(key_points_tmpl, key_points_dst, descriptors_tmpl, descriptors_dst,
                                        good_matches))
            return 0;
    } else if (descriptor_type == CornerDescriptorType::SIFT) {
        if (!calcGoodMatchesByFLANNMatcher(key_points_tmpl, key_points_dst, descriptors_tmpl, descriptors_dst,
                                           good_matches))
            return 0;
    }

    return calcTemplatePositionByMatches(image_tmpl, image_dst, position, scale, angle, key_points_tmpl,
                                         key_points_dst, good_matches);
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
