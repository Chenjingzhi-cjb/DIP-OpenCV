#include <iostream>
#include "head.h"


int main() {
    Mat image = imread("../image/lena.png");

    namedWindow("img", WINDOW_AUTOSIZE);
    imshow("img", image);
    waitKey(0);

    return 0;
}

