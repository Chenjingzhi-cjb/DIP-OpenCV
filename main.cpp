#include <iostream>
#include "head.h"


int main() {
    Mat image = imread(R"(../image/building-600by600.tif)");

    namedWindow("img", WINDOW_AUTOSIZE);
    imshow("img", image);
    waitKey(0);

    return 0;
}

