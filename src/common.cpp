#include "common.h"


void printImageData(Mat &image, Size shrink_size, int preview_unit) {
    if (image.empty()) {
        throw invalid_argument("printImageData(): Input image is empty!");
    }

    resize(image, image, shrink_size);

    cout << "-------------------------------- Image Data --------------------------------" << endl;

    cout << "Size: " << image.size() << endl;
    cout << "Depth: " << image.depth() << endl;
    cout << "Channels: " << image.channels() << endl;

    Scalar sum_value = sum(image);
    cout << "Sum value: " << sum_value << endl;
    Scalar mean_value = mean(image);
    cout << "Mean value: " << mean_value << endl;

    cout << "Image preview: " << endl;
    for (int i = 0; i < preview_unit; i++) {
        cout << "  ";
        for (int j = 0; j < preview_unit; j++) {
            int m = image.at<uchar>(i, j);
            cout << m << "\t";
        }
        for (int j = 0; j < preview_unit; j++) {
            cout << "..." << "\t";
        }
        for (int j = -1 * (preview_unit / 2); j <= 1 * (preview_unit / 2); j++) {
            int m = image.at<uchar>(i, image.cols / 2 + j);
            cout << m << "\t";
        }
        for (int j = 0; j < preview_unit; j++) {
            cout << "..." << "\t";
        }
        for (int j = preview_unit; j > 0; j--) {
            int m = image.at<uchar>(i, image.cols - j);
            cout << m << "\t";
        }
        cout << endl;
    }
    for (int i = 0; i < preview_unit; i++) {
        cout << "  ";
        for (int j = 0; j < (preview_unit * 5); j++) {
            cout << "..." << "\t";
        }
        cout << endl;
    }
    for (int i = -1 * (preview_unit / 2); i <= 1 * (preview_unit / 2); i++) {
        cout << "  ";
        for (int j = 0; j < preview_unit; j++) {
            int m = image.at<uchar>(image.rows / 2 + i, j);
            cout << m << "\t";
        }
        for (int j = 0; j < preview_unit; j++) {
            cout << "..." << "\t";
        }
        for (int j = -1 * (preview_unit / 2); j <= 1 * (preview_unit / 2); j++) {
            int m = image.at<uchar>(image.rows / 2 + i, image.cols / 2 + j);
            cout << m << "\t";
        }
        for (int j = 0; j < preview_unit; j++) {
            cout << "..." << "\t";
        }
        for (int j = preview_unit; j > 0; j--) {
            int m = image.at<uchar>(image.rows / 2 + i, image.cols - j);
            cout << m << "\t";
        }
        cout << endl;
    }
    for (int i = 0; i < preview_unit; i++) {
        cout << "  ";
        for (int j = 0; j < (preview_unit * 5); j++) {
            cout << "..." << "\t";
        }
        cout << endl;
    }
    for (int i = preview_unit; i > 0; i--) {
        cout << "  ";
        for (int j = 0; j < preview_unit; j++) {
            int m = image.at<uchar>(image.rows - i, j);
            cout << m << "\t";
        }
        for (int j = 0; j < preview_unit; j++) {
            cout << "..." << "\t";
        }
        for (int j = -1 * (preview_unit / 2); j <= 1 * (preview_unit / 2); j++) {
            int m = image.at<uchar>(image.rows - i, image.cols / 2 + j);
            cout << m << "\t";
        }
        for (int j = 0; j < preview_unit; j++) {
            cout << "..." << "\t";
        }
        for (int j = preview_unit; j > 0; j--) {
            int m = image.at<uchar>(image.rows - i, image.cols - j);
            cout << m << "\t";
        }
        cout << endl;
    }

    cout << "----------------------------------------------------------------------------" << endl;
}

void videoTraverse(VideoCapture &video) {
    if (!video.isOpened()) {
        throw invalid_argument("videoTraverse(): Video loading error!");
    }

    // 获取视频信息
    int frame_width = static_cast<int>(video.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(video.get(CAP_PROP_FRAME_HEIGHT));
    int frame_count = static_cast<int>(video.get(CAP_PROP_FRAME_COUNT));
    cout << "Video Info: " << endl;
    cout << "  frame width: " << frame_width << endl;
    cout << "  frame height: " << frame_height << endl;
    cout << "  frame count: " << frame_count << endl;

    // 遍历帧
    for (int i = 0; i < frame_count; i++) {
        Mat frame;
        video >> frame;
        // ...
    }
}
