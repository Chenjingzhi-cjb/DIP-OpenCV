#include "image_segmentation.h"


void pointDetectLaplaceKernel(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("linearSpatialFilter(): Input image is empty!");
    }

    // 孤立点检测的拉普拉斯核
     Mat kernel = (Mat_<char>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);

    filter2D(src, dst, src.depth(), kernel);
}

void lineDetectLaplaceKernel(Mat &src, Mat &dst, int line_type) {
    if (src.empty()) {
        throw invalid_argument("linearSpatialFilter(): Input image is empty!");
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

void keyPointDetectSurf(Mat &src, Mat &dst, double hessianThreshold, int nOctaves, int nOctaveLayers, bool extended,
                        bool upright) {
    if (src.empty()) {
        throw invalid_argument("keyPointDetectSurf(): Input image is empty!");
    }

    Mat temp;
    if (src.channels() == 1) {
        src.copyTo(temp);
    } else if (src.channels() == 3) {
        bgrToGray(src, temp);
    } else {
        return;
    }

    // SURF 特征检测
    Ptr<SURF> detector = SURF::create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
    vector<KeyPoint> key_points;
    detector->detect(temp, key_points, Mat());

    // 绘制关键点
    drawKeypoints(temp, key_points, temp, Scalar::all(-1));  // Scalar::all(-1) 表示每次随机选取颜色

    temp.copyTo(dst);
}