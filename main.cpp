#include <windows.h>

#include <iostream>
#include "head.h"


int main() {
//    Mat image = imread(R"(../image/building-600by600.tif)");
//
//    namedWindow("img", WINDOW_AUTOSIZE);
//    imshow("img", image);
//    waitKey(0);

    Mat image = imread(R"(../image/fa6628de.png)");
    Mat image_tmpl = imread(R"(../image/fa6628de-sub.png)");

    bgrToGray(image, image);
    bgrToGray(image_tmpl, image_tmpl);

    cv::Point2d position;
    double scale, angle;

//    calcTemplatePosition(image_tmpl, image, position, true);
//    calcTemplatePositionORB(image_tmpl, image, position, &scale, &angle);
    calcTemplatePositionCorner(image_tmpl, image, position, &scale, &angle, CornerDescriptorType::ORB);
//    calcTemplatePositionCorner(image_tmpl, image, position, &scale, &angle, CornerDescriptorType::SIFT);

    std::cout << position << std::endl;

    return 0;
}

