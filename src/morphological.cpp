#include "morphological.h"


void grayToBinary(const cv::Mat &src, cv::Mat &dst, double thresh, double maxval, int type) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::threshold(src, dst, thresh, maxval, type);
}

uchar getBinaryMaxval(const cv::Mat &src) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            uchar m = src.at<uchar>(r, c);
            if (m != 0) {
                return m;
            }
        }
    }

    return 0;
}

void binaryInvert(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::Mat temp = getBinaryMaxval(src) - src;

    temp.copyTo(dst);
}

void morphologyErode(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::erode(src, dst, kernel);
    // morphologyEx(src, dst, MORPH_ERODE, kernel);
}

void morphologyDilate(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::dilate(src, dst, kernel);
    // morphologyEx(src, dst, MORPH_DILATE, kernel);
}

void morphologyOpen(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::morphologyEx(src, dst, cv::MORPH_OPEN, kernel);
}

void morphologyClose(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::morphologyEx(src, dst, cv::MORPH_CLOSE, kernel);
}

void morphologyHMT(const cv::Mat &src, cv::Mat &dst, const cv::Mat &fore_kernel, const cv::Mat &back_kernel) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::morphologyEx(src, dst, cv::MORPH_HITMISS, fore_kernel - back_kernel);
}

void morphologyGradient(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::morphologyEx(src, dst, cv::MORPH_GRADIENT, kernel);
}

void morphologyTophat(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::morphologyEx(src, dst, cv::MORPH_TOPHAT, kernel);
}

void morphologyBlackhat(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::morphologyEx(src, dst, cv::MORPH_BLACKHAT, kernel);
}

void boundaryExtract(const cv::Mat &src, cv::Mat &dst, int size) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (size <= 0) {
        THROW_ARG_ERROR("Invalid `size`. You should make sure `size > 0`.");
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, size));

    cv::Mat image_erode;
    morphologyErode(src, image_erode, kernel);

    cv::subtract(src, image_erode, dst);
}

void holeFill(const cv::Mat &src, cv::Mat &dst, const cv::Mat &start) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (start.empty()) {
        THROW_ARG_ERROR("Input start image is empty.");
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    cv::Mat src_complement;
    binaryInvert(src, src_complement);

    // 执行孔洞填充算法
    cv::Mat temp = start.clone();
    cv::Mat temp_last;
    do {
        temp_last = temp.clone();
        morphologyDilate(temp, temp, kernel);  // 膨胀
        cv::bitwise_and(temp, src_complement, temp);  // 交集
    } while (cv::countNonZero(temp_last != temp));

    cv::bitwise_or(src, temp, temp);  // 并集

    temp.copyTo(dst);
}

int extractConnected(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::Mat I = src.clone();
    cv::Mat C = cv::Mat::zeros(I.size(), CV_8UC1);  // 记录结果
    int nc = 0;

    int p_i = 0, p_j = 0;
    while (true) {
        // 查找连通分量起点
        cv::Mat Z = cv::Mat::zeros(I.size(), CV_8UC1);
        for (; p_i < I.rows; p_i++) {
            for (; p_j < I.cols; p_j++) {
                int m = I.at<uchar>(p_i, p_j);
                if (m != 0) {
                    Z.at<uchar>(p_i, p_j) = m;
                    break;
                }
            }
            if (p_j != I.cols) {
                break;
            } else {
                p_j = 0;
            }
        }

        if (p_i == I.rows) break;

        // 执行提取连通分量算法
        cv::Mat Z_last;
        do {
            Z_last = Z.clone();
            morphologyDilate(Z, Z, kernel);  // 膨胀
            cv::bitwise_and(Z, I, Z);  // 交集
        } while (cv::countNonZero(Z_last != Z));

        // 记录连通分量
        nc += 1;
        for (int i = 0; i < Z.rows; i++) {
            for (int j = 0; j < Z.cols; j++) {
                int m = Z.at<uchar>(i, j);
                if (m != 0) {
                    C.at<uchar>(i, j) = nc;
                }
            }
        }

        cv::subtract(I, Z, I);
    }

    C.copyTo(dst);

    return nc;
}

void erodeReconstruct(const cv::Mat &src, const cv::Mat &tmpl, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if (tmpl.empty()) {
        THROW_ARG_ERROR("Input template image is empty.");
    }
    if (tmpl.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input template image type must be CV_8UC1.");
    }
    if (src.size() != tmpl.size()) {
        THROW_ARG_ERROR("The size of src and the size of tmpl must be the same.");
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // 循环执行测地腐蚀
    cv::Mat temp = src.clone();
    cv::Mat temp_last;
    do {
        temp_last = temp.clone();
        morphologyErode(temp, temp, kernel);  // 腐蚀
        // bitwise_or(temp, tmpl, temp);  // 并集
        cv::max(temp, tmpl, temp);  // 逐点最大算子
    } while (cv::countNonZero(temp_last != temp));

    temp.copyTo(dst);
}

void dilateReconstruct(const cv::Mat &src, const cv::Mat &tmpl, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }
    if (tmpl.empty()) {
        THROW_ARG_ERROR("Input template image is empty.");
    }
    if (tmpl.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input template image type must be CV_8UC1.");
    }
    if (src.size() != tmpl.size()) {
        THROW_ARG_ERROR("The size of src and the size of tmpl must be the same.");
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    // 循环执行测地膨胀
    cv::Mat temp = src.clone();
    cv::Mat temp_last;
    do {
        temp_last = temp.clone();
        morphologyDilate(temp, temp, kernel);  // 膨胀
        // bitwise_and(temp, tmpl, temp);  // 交集
        cv::min(temp, tmpl, temp);  // 逐点最小算子
    } while (cv::countNonZero(temp_last != temp));

    temp.copyTo(dst);
}

void openReconstruct(const cv::Mat &src, cv::Mat &dst, const cv::Mat &erode_kernel, int erode_times) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    // n 次腐蚀
    cv::Mat temp = src.clone();
    for (int i = 0; i < erode_times; i++) {
        morphologyErode(temp, temp, erode_kernel);
    }

    // 膨胀重建
    dilateReconstruct(temp, src, temp);

    temp.copyTo(dst);
}

void closeReconstruct(const cv::Mat &src, cv::Mat &dst, const cv::Mat &dilate_kernel, int dilate_times) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    // n 次膨胀
    cv::Mat temp = src.clone();
    for (int i = 0; i < dilate_times; i++) {
        morphologyDilate(temp, temp, dilate_kernel);
    }

    // 腐蚀重建
    erodeReconstruct(temp, src, temp);

    temp.copyTo(dst);
}

void tophatReconstruct(const cv::Mat &src, cv::Mat &dst, const cv::Mat &erode_kernel, int erode_times) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::Mat temp;
    openReconstruct(src, temp, erode_kernel, erode_times);

    cv::subtract(src, temp, dst);
}

void blackhatReconstruct(const cv::Mat &src, cv::Mat &dst, const cv::Mat &dilate_kernel, int dilate_times) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC1) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC1.");
    }

    cv::Mat temp;
    closeReconstruct(src, temp, dilate_kernel, dilate_times);

    cv::subtract(src, temp, dst);
}

void holeFill(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    uchar max_value = getBinaryMaxval(src);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::Mat src_complement;
    binaryInvert(src, src_complement);

    cv::Mat F = cv::Mat::zeros(src.size(), CV_8UC1);
    for (int col = 0; col < src.cols; col++) {
        if (src.at<uchar>(0, col) == 0) {
            F.at<uchar>(0, col) = max_value;
        }
    }
    for (int row = 1; row < src.rows - 1; row++) {
        if (src.at<uchar>(row, 0) == 0) {
            F.at<uchar>(row, 0) = max_value;
        }
        if (src.at<uchar>(row, src.cols - 1) == 0) {
            F.at<uchar>(row, src.cols - 1) = max_value;
        }
    }
    for (int col = 0; col < src.cols; col++) {
        if (src.at<uchar>(src.rows - 1, col) == 0) {
            F.at<uchar>(src.rows - 1, col) = max_value;
        }
    }

    cv::Mat temp;
    dilateReconstruct(F, src_complement, temp);

    binaryInvert(temp, temp);

    temp.copyTo(dst);
}

void borderClear(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::Mat F = cv::Mat::zeros(src.size(), CV_8UC1);
    for (int col = 0; col < src.cols; col++) {
        uchar m = src.at<uchar>(0, col);
        if (m != 0) {
            F.at<uchar>(0, col) = m;
        }
    }
    for (int row = 1; row < src.rows - 1; row++) {
        uchar m = src.at<uchar>(row, 0);
        if (m != 0) {
            F.at<uchar>(row, 0) = m;
        }
        m = src.at<uchar>(row, src.cols - 1);
        if (m != 0) {
            F.at<uchar>(row, src.cols - 1) = m;
        }
    }
    for (int col = 0; col < src.cols; col++) {
        uchar m = src.at<uchar>(src.rows - 1, col);
        if (m != 0) {
            F.at<uchar>(src.rows - 1, col) = m;
        }
    }

    cv::Mat temp;
    dilateReconstruct(F, src, temp);

    cv::subtract(src, temp, temp);

    temp.copyTo(dst);
}
