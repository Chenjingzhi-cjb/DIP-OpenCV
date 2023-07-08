#include <iostream>
#include "head.h"


int main() {
    Mat image = imread("../image/lena.png", IMREAD_GRAYSCALE);

    Mat dst1;
    DWT(image, dst1, "sym2", 3);

    Mat dst2;
    IDWT(dst1, dst2, "sym2", 3);

    normalize(image, image, 0, 255, NORM_MINMAX, CV_8U);
    normalize(dst1, dst1, 0, 255, NORM_MINMAX, CV_8U);
    normalize(dst2, dst2, 0, 255, NORM_MINMAX, CV_8U);

    namedWindow("img", WINDOW_AUTOSIZE);
    imshow("img", image);
    namedWindow("dst1", WINDOW_AUTOSIZE);
    imshow("dst1", dst1);
    namedWindow("dst2", WINDOW_AUTOSIZE);
    imshow("dst2", dst2);
    waitKey(0);

    return 0;
}

