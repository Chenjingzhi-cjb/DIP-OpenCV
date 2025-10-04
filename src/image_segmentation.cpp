#include "image_segmentation.h"


void pointDetectLaplaceKernel(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("pointDetectLaplaceKernel(): Input image is empty!");
    }

    // 孤立点检测的拉普拉斯核
    Mat kernel = (Mat_<char>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);

    filter2D(src, dst, src.depth(), kernel);
}

void lineDetectLaplaceKernel(Mat &src, Mat &dst, int line_type) {
    if (src.empty()) {
        throw invalid_argument("lineDetectLaplaceKernel(): Input image is empty!");
    }

    // 线检测的拉普拉斯核
    Mat kernel;

    if (line_type == 1) {
        kernel = (Mat_<char>(3, 3) << -1, -1, -1, 2, 2, 2, -1, -1, -1);  // 水平
    } else if (line_type == 2) {
        kernel = (Mat_<char>(3, 3) << 2, -1, -1, -1, 2, -1, -1, -1, 2);  // 西北-东南 方向
    } else if (line_type == 3) {
        kernel = (Mat_<char>(3, 3) << -1, 2, -1, -1, 2, -1, -1, 2, -1);  // 垂直
    } else if (line_type == 4) {
        kernel = (Mat_<char>(3, 3) << -1, -1, 2, -1, 2, -1, 2, -1, -1);  // 东北-西南 方向
    } else {
        return;
    }

    filter2D(src, dst, src.depth(), kernel);
}

void lineDetectHough(Mat &src, Mat &dst, double rho, double theta, int threshold, double srn, double stn,
                     double min_theta, double max_theta) {
    if (src.empty()) {
        throw invalid_argument("lineDetectHough(): Input image is empty!");
    }

    vector<Vec2f> lines;
    HoughLines(src, lines, rho, theta, threshold, srn, stn, min_theta, max_theta);

    Mat temp = src.clone();
    for (auto &l: lines) {
        float line_rho = l[0], line_theta = l[1];
        double a = cos(line_theta), b = sin(line_theta);
        double x0 = a * line_rho, y0 = b * line_rho;

        Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
        line(temp, pt1, pt2, Scalar(255, 0, 0), 3);
    }

    temp.copyTo(dst);
}

void lineSegmentDetectHough(Mat &src, Mat &dst, double rho, double theta, int threshold, double minLineLength,
                            double maxLineGap) {
    if (src.empty()) {
        throw invalid_argument("lineSegmentDetectHough(): Input image is empty!");
    }

    vector<Vec4i> lines;
    HoughLinesP(src, lines, rho, theta, threshold, minLineLength, maxLineGap);

    Mat temp = src.clone();
    for (auto l: lines) {
        line(temp, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 3);
    }

    temp.copyTo(dst);
}

void circleDetectHough(Mat &src, Mat &dst, int method, double dp, double minDist, double param1, double param2,
                       int minRadius, int maxRadius) {
    if (src.empty()) {
        throw invalid_argument("circleDetectHough(): Input image is empty!");
    }

    vector<Vec3f> circles;
    HoughCircles(src, circles, method, dp, minDist, param1, param2, minRadius, maxRadius);

    Mat temp = src.clone();
    for (const auto &c: circles) {
        Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);

        // 绘制圆心
        circle(temp, center, 3, Scalar(0, 255, 0), FILLED);
        // 绘制圆轮廓
        circle(temp, center, radius, Scalar(255, 0, 0), 3);
    }

    temp.copyTo(dst);
}

void cornerDetectHarris(Mat &src, Mat &dst, int threshold, int blockSize, int ksize, double k, int borderType) {
    if (src.empty()) {
        throw invalid_argument("cornerDetectHarris(): Input image is empty!");
    }

    Mat temp;
    if (src.channels() == 1) {
        src.copyTo(temp);
    } else if (src.channels() == 3) {
        bgrToGray(src, temp);
    } else {
        return;
    }

    // 计算 Harris 值
    Mat harris_value;
    cornerHarris(temp, harris_value, blockSize, ksize, k, borderType);

    // 归一化
    normalize(harris_value, harris_value, 0, 255, NORM_MINMAX, CV_8UC1);

    // 阈值处理 并 绘制角点
    src.copyTo(temp);
    if (temp.channels() == 1) {
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.cols; j++) {
                if (harris_value.at<uchar>(i, j) > threshold) {
                    circle(temp, Point(j, i), 2, Scalar(255), 2, 8, 0);
                }
            }
        }
    } else if (temp.channels() == 3) {
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.cols; j++) {
                if (harris_value.at<uchar>(i, j) > threshold) {
                    circle(temp, Point(j, i), 2, Scalar(0, 255, 0), 2, 8, 0);
                }
            }
        }
    }

    temp.copyTo(dst);
}

void cornerDetectShiTomasi(Mat &src, Mat &dst, int maxCorners, double qualityLevel, double minDistance, InputArray mask,
                           int blockSize) {
    if (src.empty()) {
        throw invalid_argument("cornerDetectShiTomasi(): Input image is empty!");
    }

    Mat temp;
    if (src.channels() == 1) {
        src.copyTo(temp);
    } else if (src.channels() == 3) {
        bgrToGray(src, temp);
    } else {
        return;
    }

    // 计算角点坐标
    vector<Point2f> corners;
    goodFeaturesToTrack(temp, corners, maxCorners, qualityLevel, minDistance, mask, blockSize);

    // 绘制角点
    src.copyTo(temp);
    if (temp.channels() == 1) {
        for (auto &corner: corners) {
            circle(temp, corner, 2, Scalar(255), 2, 8, 0);
        }
    } else if (temp.channels() == 3) {
        for (auto &corner: corners) {
            circle(temp, corner, 2, Scalar(0, 255, 0), 2, 8, 0);
        }
    }

    temp.copyTo(dst);
}

void cornerDetectSubPixel(Mat &src, Mat &dst, int maxCorners, double qualityLevel, double minDistance, Size winSize,
                          Size zeroZone, TermCriteria criteria, InputArray mask, int blockSize, bool useHarrisDetector,
                          double k) {
    if (src.empty()) {
        throw invalid_argument("cornerDetectSubPixel(): Input image is empty!");
    }

    Mat temp;
    if (src.channels() == 1) {
        src.copyTo(temp);
    } else if (src.channels() == 3) {
        bgrToGray(src, temp);
    } else {
        return;
    }

    // 计算角点坐标
    vector<Point2f> corners;
    goodFeaturesToTrack(temp, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);

    // 计算亚像素级角点坐标
    cornerSubPix(temp, corners, winSize, zeroZone, criteria);

    // 绘制角点
    src.copyTo(temp);
    if (temp.channels() == 1) {
        for (auto &corner: corners) {
            circle(temp, corner, 2, Scalar(255), 2, 8, 0);
        }
    } else if (temp.channels() == 3) {
        for (auto &corner: corners) {
            circle(temp, corner, 2, Scalar(0, 255, 0), 2, 8, 0);
        }
    }

    temp.copyTo(dst);
}
