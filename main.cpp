#include <iostream>
#include "head.h"


int main() {
    Mat image = imread(R"(..\image\lena.png)");

    Mat image_gray;
    bgrToGray(image, image_gray);

    Mat dst;
    Mat ideal_kernel = bwLowFrequencyKernel(image_gray.size(), 20, 1);
    smoothFrequencyFilter(image_gray, dst, ideal_kernel);

    namedWindow("gray", WINDOW_AUTOSIZE);
    imshow("gray", image_gray);
    namedWindow("dst", WINDOW_AUTOSIZE);
    imshow("dst", dst);
    waitKey(0);

    return 0;
}

