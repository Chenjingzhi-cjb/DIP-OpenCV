#include <iostream>
#include "head.h"


int main() {
    Mat image = imread(R"(..\image\test-pattern.tif)");

    Mat image_gray;
    bgrToGray(image, image_gray);

    Mat dst;
    addNoiseSaltPepper(image_gray, dst, 0.05);

    Mat ht_src = grayHistogram(image_gray);
    Mat ht_dst = grayHistogram(dst);

    namedWindow("gray", WINDOW_AUTOSIZE);
    imshow("gray", image_gray);
    namedWindow("dst", WINDOW_AUTOSIZE);
    imshow("dst", dst);
    namedWindow("ht_src", WINDOW_AUTOSIZE);
    imshow("ht_src", ht_src);
    namedWindow("ht_dst", WINDOW_AUTOSIZE);
    imshow("ht_dst", ht_dst);
    waitKey(0);

    return 0;
}

