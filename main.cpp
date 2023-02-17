#include <iostream>
#include "head.h"


int main() {
    Mat img = imread("../image/lena.png");

    Mat img_gray;
    bgrToGray(img, img_gray);

    Mat dst;
    spatialToFrequency(img_gray, dst);

    namedWindow("dst", WINDOW_AUTOSIZE);
    imshow("dst", dst);
    waitKey(0);

    return 0;
}

