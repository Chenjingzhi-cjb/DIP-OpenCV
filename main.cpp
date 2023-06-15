#include <iostream>
#include "head.h"


int main() {
    Mat src = (Mat_<uchar>(6, 6)
            << 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 1, 0,
            1, 1, 1, 0, 1, 0,
            0, 1, 0, 0, 0, 0,
            0, 1, 0, 1, 1, 1,
            0, 0, 0, 0, 0, 0);

    Mat dst;
    extractConnected(src, dst);

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            int m = src.at<uchar>(i, j);
            std::cout << m << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            int m = dst.at<uchar>(i, j);
            std::cout << m << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return 0;
}

