#include <iostream>
#include "head.h"


int main() {
    Mat image = imread(R"(..\image\hidden-symbols.tif)");

    Mat image_gray;
    bgrToGray(image, image_gray);

    Mat dst;
    localEqualizeHist(image_gray, dst, 40, Size(64, 64));

    namedWindow("gray", WINDOW_AUTOSIZE);
    imshow("gray", image_gray);
    namedWindow("dst", WINDOW_AUTOSIZE);
    imshow("dst", dst);
    waitKey(0);

    return 0;
}

