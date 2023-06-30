#include "morphological.h"


void grayToBinary(Mat &src, Mat &dst, double thresh, double maxval, int type) {
    if (src.empty()) {
        throw invalid_argument("grayToBinary(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("grayToBinary(): Input src image's type error! It should be CV_8UC1!");
    }

    threshold(src, dst, thresh, maxval, type);
}

uchar getBinaryMaxval(Mat &src) {
    if (src.empty()) {
        throw invalid_argument("getBinaryMaxval(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("getBinaryMaxval(): Input src image's type error! It should be CV_8UC1!");
    }

    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            auto m = src.at<uchar>(r, c);
            if (m != 0) {
                return m;
            }
        }
    }

    return 0;
}

void binaryInvert(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("binaryInvert(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("binaryInvert(): Input src image's type error! It should be CV_8UC1!");
    }

    Mat temp = getBinaryMaxval(src) - src;

    temp.copyTo(dst);
}

void morphologyErode(Mat &src, Mat &dst, const Mat &kernel) {
    if (src.empty()) {
        throw invalid_argument("morphologyErode(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("morphologyErode(): Input src image's type error! It should be CV_8UC1!");
    }

    erode(src, dst, kernel);
    // morphologyEx(src, dst, MORPH_ERODE, kernel);
}

void morphologyDilate(Mat &src, Mat &dst, const Mat &kernel) {
    if (src.empty()) {
        throw invalid_argument("morphologyDilate(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("morphologyDilate(): Input src image's type error! It should be CV_8UC1!");
    }

    dilate(src, dst, kernel);
    // morphologyEx(src, dst, MORPH_DILATE, kernel);
}

void morphologyOpen(Mat &src, Mat &dst, const Mat &kernel) {
    if (src.empty()) {
        throw invalid_argument("morphologyOpen(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("morphologyOpen(): Input src image's type error! It should be CV_8UC1!");
    }

    morphologyEx(src, dst, MORPH_OPEN, kernel);
}

void morphologyClose(Mat &src, Mat &dst, const Mat &kernel) {
    if (src.empty()) {
        throw invalid_argument("morphologyClose(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("morphologyClose(): Input src image's type error! It should be CV_8UC1!");
    }

    morphologyEx(src, dst, MORPH_CLOSE, kernel);
}

void morphologyHMT(Mat &src, Mat &dst, const Mat &fore_kernel, const Mat &back_kernel) {
    if (src.empty()) {
        throw invalid_argument("morphologyHMT(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("morphologyHMT(): Input src image's type error! It should be CV_8UC1!");
    }

    morphologyEx(src, dst, cv::MORPH_HITMISS, fore_kernel - back_kernel);
}

void morphologyGradient(Mat &src, Mat &dst, const Mat &kernel) {
    if (src.empty()) {
        throw invalid_argument("morphologyGradient(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("morphologyGradient(): Input src image's type error! It should be CV_8UC1!");
    }

    morphologyEx(src, dst, cv::MORPH_GRADIENT, kernel);
}

void morphologyTophat(Mat &src, Mat &dst, const Mat &kernel) {
    if (src.empty()) {
        throw invalid_argument("morphologyTophat(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("morphologyTophat(): Input src image's type error! It should be CV_8UC1!");
    }

    morphologyEx(src, dst, cv::MORPH_TOPHAT, kernel);
}

void morphologyBlackhat(Mat &src, Mat &dst, const Mat &kernel) {
    if (src.empty()) {
        throw invalid_argument("morphologyBlackhat(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("morphologyBlackhat(): Input src image's type error! It should be CV_8UC1!");
    }

    morphologyEx(src, dst, cv::MORPH_BLACKHAT, kernel);
}

void boundaryExtract(Mat &src, Mat &dst, int size) {
    if (src.empty()) {
        throw invalid_argument("boundaryExtract(): Input src image is empty!");
    }

    if (size <= 0) {
        string err = R"(boundaryExtract(): Parameter Error! You should make sure "size > 0"!)";
        throw invalid_argument(err);
    }

    Mat kernel = getStructuringElement(MORPH_RECT, Size(size, size));

    Mat image_erode;
    morphologyErode(src, image_erode, kernel);

    subtract(src, image_erode, dst);
}

void holeFill(Mat &src, Mat &dst, Mat &start) {
    if (src.empty()) {
        throw invalid_argument("holeFill(): Input src image is empty!");
    }

    if (start.empty()) {
        throw invalid_argument("holeFill(): Input start image is empty!");
    }

    Mat kernel = getStructuringElement(MORPH_CROSS, Size(3, 3));

    Mat src_complement;
    binaryInvert(src, src_complement);

    // 执行孔洞填充算法
    Mat temp = start.clone();
    Mat temp_last;
    do {
        temp_last = temp.clone();
        morphologyDilate(temp, temp, kernel);  // 膨胀
        bitwise_and(temp, src_complement, temp);  // 交集
    } while (countNonZero(temp_last != temp));

    bitwise_or(src, temp, temp);  // 并集

    temp.copyTo(dst);
}

void extractConnected(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("extractConnected(): Input src image is empty!");
    }

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    Mat I = src.clone();
    Mat C = Mat::zeros(I.size(), CV_8UC1);  // 记录结果
    int nc = 0;

    int p_i = 0, p_j = 0;
    while (true) {
        // 查找连通分量起点
        Mat Z = Mat::zeros(I.size(), CV_8UC1);
        for (; p_i < I.rows; p_i++) {
            for (; p_j < I.cols; p_j++) {
                int m = I.at<uchar>(p_i, p_j);
                if (m != 0) {
                    Z.at<uchar>(p_i, p_j) = m;
                    break;
                }
            }
            if (p_j != I.rows) {
                break;
            } else {
                p_j = 0;
            }
        }

        if (p_i == I.rows) break;

        // 执行提取连通分量算法
        Mat Z_last;
        do {
            Z_last = Z.clone();
            morphologyDilate(Z, Z, kernel);  // 膨胀
            bitwise_and(Z, I, Z);  // 交集
        } while (countNonZero(Z_last != Z));

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

        subtract(I, Z, I);
    }

    C.copyTo(dst);
}

void erodeReconstruct(Mat &src, const Mat &tmpl, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("erodeReconstruct(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("erodeReconstruct(): Input src image's type error! It should be CV_8UC1!");
    }

    if (tmpl.empty()) {
        throw invalid_argument("erodeReconstruct(): Input template image is empty!");
    }

    if (tmpl.type() != CV_8UC1) {
        throw invalid_argument("erodeReconstruct(): Input template image's type error! It should be CV_8UC1!");
    }

    if (src.size() != tmpl.size()) {
        throw invalid_argument("erodeReconstruct(): The size of src and the size of tmpl must be the same!");
    }

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    // 循环执行测地腐蚀
    Mat temp = src.clone();
    Mat temp_last;
    do {
        temp_last = temp.clone();
        morphologyErode(temp, temp, kernel);  // 腐蚀
        // bitwise_or(temp, tmpl, temp);  // 并集
        cv::max(temp, tmpl, temp);  // 逐点最大算子
    } while (countNonZero(temp_last != temp));

    temp.copyTo(dst);
}

void dilateReconstruct(Mat &src, const Mat &tmpl, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("dilateReconstruct(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("dilateReconstruct(): Input src image's type error! It should be CV_8UC1!");
    }

    if (tmpl.empty()) {
        throw invalid_argument("dilateReconstruct(): Input template image is empty!");
    }

    if (tmpl.type() != CV_8UC1) {
        throw invalid_argument("dilateReconstruct(): Input template image's type error! It should be CV_8UC1!");
    }

    if (src.size() != tmpl.size()) {
        throw invalid_argument("dilateReconstruct(): The size of src and the size of tmpl must be the same!");
    }

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    // 循环执行测地膨胀
    Mat temp = src.clone();
    Mat temp_last;
    do {
        temp_last = temp.clone();
        morphologyDilate(temp, temp, kernel);  // 膨胀
        // bitwise_and(temp, tmpl, temp);  // 交集
        cv::min(temp, tmpl, temp);  // 逐点最小算子
    } while (countNonZero(temp_last != temp));

    temp.copyTo(dst);
}

void openReconstruct(Mat &src, Mat &dst, const Mat &erode_kernel, int erode_times) {
    if (src.empty()) {
        throw invalid_argument("openReconstruct(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("openReconstruct(): Input src image's type error! It should be CV_8UC1!");
    }

    // n 次腐蚀
    Mat temp = src.clone();
    for (int i = 0; i < erode_times; i++) {
        morphologyErode(temp, temp, erode_kernel);
    }

    // 膨胀重建
    dilateReconstruct(temp, src, temp);

    temp.copyTo(dst);
}

void closeReconstruct(Mat &src, Mat &dst, const Mat &dilate_kernel, int dilate_times) {
    if (src.empty()) {
        throw invalid_argument("closeReconstruct(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("closeReconstruct(): Input src image's type error! It should be CV_8UC1!");
    }

    // n 次膨胀
    Mat temp = src.clone();
    for (int i = 0; i < dilate_times; i++) {
        morphologyDilate(temp, temp, dilate_kernel);
    }

    // 腐蚀重建
    erodeReconstruct(temp, src, temp);

    temp.copyTo(dst);
}

void tophatReconstruct(Mat &src, Mat &dst, const Mat &erode_kernel, int erode_times) {
    if (src.empty()) {
        throw invalid_argument("tophatReconstruct(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("tophatReconstruct(): Input src image's type error! It should be CV_8UC1!");
    }

    Mat temp;
    openReconstruct(src, temp, erode_kernel, erode_times);

    subtract(src, temp, dst);
}

void blackhatReconstruct(Mat &src, Mat &dst, const Mat &dilate_kernel, int dilate_times) {
    if (src.empty()) {
        throw invalid_argument("blackhatReconstruct(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("blackhatReconstruct(): Input src image's type error! It should be CV_8UC1!");
    }

    Mat temp;
    closeReconstruct(src, temp, dilate_kernel, dilate_times);

    subtract(src, temp, dst);
}

void holeFill(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("holeFill(): Input src image is empty!");
    }

    int max_value = getBinaryMaxval(src);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    Mat src_complement;
    binaryInvert(src, src_complement);

    Mat F = Mat::zeros(src.size(), CV_8UC1);
    for (int col = 0; col < src.cols; col++) {
        if (src.at<uchar>(0, col) == 0) F.at<uchar>(0, col) = max_value;
    }
    for (int row = 1; row < src.rows - 1; row++) {
        if (src.at<uchar>(row, 0) == 0) F.at<uchar>(row, 0) = max_value;
        if (src.at<uchar>(row, src.cols - 1) == 0) F.at<uchar>(row, src.cols - 1) = max_value;
    }
    for (int col = 0; col < src.cols; col++) {
        if (src.at<uchar>(src.rows - 1, col) == 0) F.at<uchar>(src.rows - 1, col) = max_value;
    }

    Mat temp;
    dilateReconstruct(F, src_complement, temp);

    binaryInvert(temp, temp);

    temp.copyTo(dst);
}

void borderClear(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("borderClear(): Input src image is empty!");
    }

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    Mat F = Mat::zeros(src.size(), CV_8UC1);
    for (int col = 0; col < src.cols; col++) {
        auto m = src.at<uchar>(0, col);
        if (m != 0) F.at<uchar>(0, col) = m;
    }
    for (int row = 1; row < src.rows - 1; row++) {
        auto m = src.at<uchar>(row, 0);
        if (m != 0) F.at<uchar>(row, 0) = m;
        m = src.at<uchar>(row, src.cols - 1);
        if (m != 0) F.at<uchar>(row, src.cols - 1) = m;
    }
    for (int col = 0; col < src.cols; col++) {
        auto m = src.at<uchar>(src.rows - 1, col);
        if (m != 0) F.at<uchar>(src.rows - 1, col) = m;
    }

    Mat temp;
    dilateReconstruct(F, src, temp);

    subtract(src, temp, temp);

    temp.copyTo(dst);
}
