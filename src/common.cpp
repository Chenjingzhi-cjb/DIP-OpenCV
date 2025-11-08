#include "common.h"


void printImageData(const cv::Mat &image, cv::Size shrink_size, int preview_unit) {
    if (image.empty()) {
        THROW_ARG_ERROR("The input image is empty.");
    }

    cv::Mat temp;
    cv::resize(image, temp, shrink_size);

    std::cout << "-------------------------------- Image Data --------------------------------" << std::endl;

    std::cout << "Size: " << temp.size() << std::endl;
    std::cout << "Depth: " << temp.depth() << std::endl;
    std::cout << "Channels: " << temp.channels() << std::endl;

    cv::Scalar sum_value = cv::sum(temp);
    std::cout << "Sum value: " << sum_value << std::endl;
    cv::Scalar mean_value = cv::mean(temp);
    std::cout << "Mean value: " << mean_value << std::endl;

    std::cout << "Image preview: " << std::endl;
    for (int i = 0; i < preview_unit; i++) {
        std::cout << "  ";
        for (int j = 0; j < preview_unit; j++) {
            int m = temp.at<uchar>(i, j);
            std::cout << m << "\t";
        }
        for (int j = 0; j < preview_unit; j++) {
            std::cout << "..." << "\t";
        }
        for (int j = -1 * (preview_unit / 2); j <= 1 * (preview_unit / 2); j++) {
            int m = temp.at<uchar>(i, temp.cols / 2 + j);
            std::cout << m << "\t";
        }
        for (int j = 0; j < preview_unit; j++) {
            std::cout << "..." << "\t";
        }
        for (int j = preview_unit; j > 0; j--) {
            int m = temp.at<uchar>(i, temp.cols - j);
            std::cout << m << "\t";
        }
        std::cout << std::endl;
    }
    for (int i = 0; i < preview_unit; i++) {
        std::cout << "  ";
        for (int j = 0; j < (preview_unit * 5); j++) {
            std::cout << "..." << "\t";
        }
        std::cout << std::endl;
    }
    for (int i = -1 * (preview_unit / 2); i <= 1 * (preview_unit / 2); i++) {
        std::cout << "  ";
        for (int j = 0; j < preview_unit; j++) {
            int m = temp.at<uchar>(temp.rows / 2 + i, j);
            std::cout << m << "\t";
        }
        for (int j = 0; j < preview_unit; j++) {
            std::cout << "..." << "\t";
        }
        for (int j = -1 * (preview_unit / 2); j <= 1 * (preview_unit / 2); j++) {
            int m = temp.at<uchar>(temp.rows / 2 + i, temp.cols / 2 + j);
            std::cout << m << "\t";
        }
        for (int j = 0; j < preview_unit; j++) {
            std::cout << "..." << "\t";
        }
        for (int j = preview_unit; j > 0; j--) {
            int m = temp.at<uchar>(temp.rows / 2 + i, temp.cols - j);
            std::cout << m << "\t";
        }
        std::cout << std::endl;
    }
    for (int i = 0; i < preview_unit; i++) {
        std::cout << "  ";
        for (int j = 0; j < (preview_unit * 5); j++) {
            std::cout << "..." << "\t";
        }
        std::cout << std::endl;
    }
    for (int i = preview_unit; i > 0; i--) {
        std::cout << "  ";
        for (int j = 0; j < preview_unit; j++) {
            int m = temp.at<uchar>(temp.rows - i, j);
            std::cout << m << "\t";
        }
        for (int j = 0; j < preview_unit; j++) {
            std::cout << "..." << "\t";
        }
        for (int j = -1 * (preview_unit / 2); j <= 1 * (preview_unit / 2); j++) {
            int m = temp.at<uchar>(temp.rows - i, temp.cols / 2 + j);
            std::cout << m << "\t";
        }
        for (int j = 0; j < preview_unit; j++) {
            std::cout << "..." << "\t";
        }
        for (int j = preview_unit; j > 0; j--) {
            int m = temp.at<uchar>(temp.rows - i, temp.cols - j);
            std::cout << m << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << "----------------------------------------------------------------------------" << std::endl;
}

void videoTraverse(cv::VideoCapture &video) {
    if (!video.isOpened()) {
        THROW_ARG_ERROR("Video loading error!");
    }

    // 获取视频信息
    int frame_width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frame_count = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Video Info: " << std::endl;
    std::cout << "  frame width: " << frame_width << std::endl;
    std::cout << "  frame height: " << frame_height << std::endl;
    std::cout << "  frame count: " << frame_count << std::endl;

    // 遍历帧
    for (int i = 0; i < frame_count; i++) {
        cv::Mat frame;
        video >> frame;
        // ...
    }
}
