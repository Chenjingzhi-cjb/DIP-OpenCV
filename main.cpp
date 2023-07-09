#include <iostream>
#include "head.h"


int main() {
    Mat image = imread("../image/chestXray.tif", IMREAD_GRAYSCALE);

    namedWindow("img", WINDOW_AUTOSIZE);
    imshow("img", image);
    waitKey(0);

    return 0;
}

