#include <iostream>
#include "head.h"


int main() {
    Mat image = imread("../image/text-touching-border-half.tif", IMREAD_GRAYSCALE);

    Mat image_binary;
    threshold(image, image_binary, 127, 1, THRESH_BINARY);

    Mat dst;
    borderClear(image_binary, dst);

    image_binary.convertTo(image_binary, image_binary.depth(), 255, 0);
    dst.convertTo(dst, dst.depth(), 255, 0);

    namedWindow("img", WINDOW_NORMAL);
    imshow("img", image_binary);
    namedWindow("dst", WINDOW_NORMAL);
    imshow("dst", dst);
    waitKey(0);

    return 0;
}

