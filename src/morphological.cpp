#include "morphological.h"


void morphologyErode(Mat &src, Mat &dst, const Mat &kernel) {
    if (src.empty()) {
        throw invalid_argument("morphologyErode(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("morphologyErode(): Input image's type error! It should be CV_8UC1");
    }

    if ((kernel.rows <= 0) || (kernel.rows % 2 == 0) || (kernel.cols <= 0) || (kernel.cols % 2 == 0)) {
        string err = R"(morphologyErode(): Parameter Error! You should make sure kernel size is positive odd number!)";
        throw invalid_argument(err);
    }

    erode(src, dst, kernel);
    // morphologyEx(src, dst, MORPH_ERODE, kernel);
}

void morphologyDilate(Mat &src, Mat &dst, const Mat &kernel) {
    if (src.empty()) {
        throw invalid_argument("morphologyDilate(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("morphologyDilate(): Input image's type error! It should be CV_8UC1");
    }

    if ((kernel.rows <= 0) || (kernel.rows % 2 == 0) || (kernel.cols <= 0) || (kernel.cols % 2 == 0)) {
        string err = R"(morphologyDilate(): Parameter Error! You should make sure kernel size is positive odd number!)";
        throw invalid_argument(err);
    }

    dilate(src, dst, kernel);
    // morphologyEx(src, dst, MORPH_DILATE, kernel);
}

void morphologyOpen(Mat &src, Mat &dst, const Mat &kernel) {
    if (src.empty()) {
        throw invalid_argument("morphologyOpen(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("morphologyOpen(): Input image's type error! It should be CV_8UC1");
    }

    if ((kernel.rows <= 0) || (kernel.rows % 2 == 0) || (kernel.cols <= 0) || (kernel.cols % 2 == 0)) {
        string err = R"(morphologyOpen(): Parameter Error! You should make sure kernel size is positive odd number!)";
        throw invalid_argument(err);
    }

    morphologyEx(src, dst, MORPH_OPEN, kernel);
}

void morphologyClose(Mat &src, Mat &dst, const Mat &kernel) {
    if (src.empty()) {
        throw invalid_argument("morphologyClose(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("morphologyClose(): Input image's type error! It should be CV_8UC1");
    }

    if ((kernel.rows <= 0) || (kernel.rows % 2 == 0) || (kernel.cols <= 0) || (kernel.cols % 2 == 0)) {
        string err = R"(morphologyClose(): Parameter Error! You should make sure kernel size is positive odd number!)";
        throw invalid_argument(err);
    }

    morphologyEx(src, dst, MORPH_CLOSE, kernel);
}

void morphologyHMT(Mat &src, Mat &dst, const Mat &fore_kernel, const Mat &back_kernel) {
    if (src.empty()) {
        throw invalid_argument("morphologyHMT(): Input src image is empty!");
    }

    if (src.type() != CV_8UC1) {
        throw invalid_argument("morphologyHMT(): Input image's type error! It should be CV_8UC1");
    }

    if ((fore_kernel.rows <= 0) || (fore_kernel.rows % 2 == 0) || (fore_kernel.cols <= 0) ||
        (fore_kernel.cols % 2 == 0)) {
        string err = R"(morphologyHMT(): Parameter Error! You should make sure fore_kernel size is positive odd number!)";
        throw invalid_argument(err);
    }

    if ((back_kernel.rows <= 0) || (back_kernel.rows % 2 == 0) || (back_kernel.cols <= 0) ||
        (back_kernel.cols % 2 == 0)) {
        string err = R"(morphologyHMT(): Parameter Error! You should make sure back_kernel size is positive odd number!)";
        throw invalid_argument(err);
    }

    morphologyEx(src, dst, cv::MORPH_HITMISS, fore_kernel - back_kernel);
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
    bitwise_not(src, src_complement);

    // 执行提取连通分量算法
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
    Mat C = Mat::zeros(I.size(), CV_8U);  // 记录结果
    int nc = 0;

    int p_i = 0, p_j = 0;
    while (true) {
        // 查找连通分量起点
        Mat Z = Mat::zeros(I.size(), CV_8U);
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
