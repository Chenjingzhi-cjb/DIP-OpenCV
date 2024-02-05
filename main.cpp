#include <iostream>
#include "head.h"


int main() {
    Mat image = imread(R"(../image/building-600by600.tif)");

    Mat dst;
//    cornerDetectHarris(image, dst, 120, 3, 3);
//    cornerDetectShiTomasi(image, dst, 120, 0.01, 10);
    cornerDetectSubPixel(image, dst, 120, 0.01, 10, Size(5, 5), Size(-1, -1),
                         TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001));

    namedWindow("img", WINDOW_AUTOSIZE);
    imshow("img", dst);
    waitKey(0);

    return 0;
}

