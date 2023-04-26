#include <iostream>
#include "head.h"


int main() {
    Mat image = imread(R"(..\image\lena.png)");

    Mat image_gray;
    bgrToGray(image, image_gray);

    namedWindow("gray", WINDOW_AUTOSIZE);
    imshow("gray", image_gray);
    waitKey(0);

    return 0;
}

