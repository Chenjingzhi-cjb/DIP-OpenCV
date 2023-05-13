#include <iostream>
#include "head.h"


int main() {
    Mat image = imread(R"(..\image\checkerboard512-shaded.tif)");

    Mat image_gray;
    bgrToGray(image, image_gray);

    Mat dst;
    shadingCorrection(image_gray, dst);

//    namedWindow("raw", WINDOW_AUTOSIZE);
//    imshow("raw", image);
    namedWindow("gray", WINDOW_AUTOSIZE);
    imshow("gray", image_gray);
    namedWindow("dst", WINDOW_AUTOSIZE);
    imshow("dst", dst);
    waitKey(0);

    return 0;
}

