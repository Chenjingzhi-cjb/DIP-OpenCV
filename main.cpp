#include <iostream>
#include "head.h"


int main() {
    Mat image = imread(R"(..\image\chestXray.tif)");

    Mat image_gray;
    bgrToGray(image, image_gray);

    Mat dst;
    Mat kernel = highFreqEmphasisKernel(image_gray.size(), 30);
    frequencyFilter(image_gray, dst, kernel);

    namedWindow("gray", WINDOW_AUTOSIZE);
    imshow("gray", image_gray);
    namedWindow("dst", WINDOW_AUTOSIZE);
    imshow("dst", dst);
    waitKey(0);

    return 0;
}

