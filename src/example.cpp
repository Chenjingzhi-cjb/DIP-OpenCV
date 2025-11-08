#include "example.h"


void localEqualizeHistExample() {
    cv::Mat image_gray = cv::imread(R"(..\image\hidden-symbols.tif)", 0);

    cv::Mat dst;
    localEqualizeHist(image_gray, dst);

    cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
    cv::imshow("src", image_gray);
    cv::namedWindow("dst", cv::WINDOW_AUTOSIZE);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}

void shadingCorrectionExample() {
    cv::Mat image_gray = cv::imread(R"(..\image\checkerboard512-shaded.tif)", 0);

    cv::Mat dst;
    shadingCorrection(image_gray, dst);

    cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
    cv::imshow("src", image_gray);
    cv::namedWindow("dst", cv::WINDOW_AUTOSIZE);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}

void domainTransformExample() {
    cv::Mat image_gray = cv::imread(R"(..\image\barbara.tif)", 0);

    cv::Mat dst_complex, dst_magnitude;
    spatialToFrequency(image_gray, dst_complex);
    splitFrequencyMagnitude(dst_complex, dst_magnitude);

    cv::Mat dst_idft = cv::Mat::zeros(image_gray.size(), image_gray.depth());
    frequencyToSpatial(dst_complex, dst_idft, image_gray.size());
    cv::normalize(dst_idft, dst_idft, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
    cv::imshow("src", image_gray);
    cv::namedWindow("dft", cv::WINDOW_AUTOSIZE);
    cv::imshow("dft", dst_magnitude);
    cv::namedWindow("idft", cv::WINDOW_AUTOSIZE);
    cv::imshow("idft", dst_idft);
    cv::waitKey(0);
}

void highFreqEmphasisExample() {
    cv::Mat image_gray = cv::imread(R"(..\image\chestXray.tif)", 0);

    cv::Mat kernel = highFreqEmphasisKernel(image_gray.size(), 70);

    cv::Mat dst;
    frequencyFilter(image_gray, dst, kernel);

    cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
    cv::imshow("src", image_gray);
    cv::namedWindow("dst", cv::WINDOW_AUTOSIZE);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}

void hsiExample() {
    cv::Mat image = cv::imread(R"(..\image\strawberries-RGB.tif)");

    cv::Mat dst;
    bgrToHsi(image, dst);

    cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
    cv::imshow("src", image);
    cv::namedWindow("dst", cv::WINDOW_AUTOSIZE);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}

void DCTExample() {
    cv::Mat image_gray = cv::imread(R"(..\image\barbara.tif)", 0);
    image_gray.convertTo(image_gray, CV_32F, 1.0 / 255);

    cv::Mat dst_dct;
    DCT(image_gray, dst_dct);

    cv::Mat dst_idct = cv::Mat::zeros(image_gray.size(), image_gray.type());
    IDCT(dst_dct, dst_idct, image_gray.size());
    cv::normalize(dst_idct, dst_idct, 0, 1, cv::NORM_MINMAX);

    cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
    cv::imshow("src", image_gray);
    cv::namedWindow("dct", cv::WINDOW_AUTOSIZE);
    cv::imshow("dct", dst_dct);
    cv::namedWindow("idct", cv::WINDOW_AUTOSIZE);
    cv::imshow("idct", dst_idct);
    cv::waitKey(0);
}

void blockDCTExample() {
    cv::Mat image_gray = cv::imread(R"(..\image\barbara.tif)", 0);
    image_gray.convertTo(image_gray, CV_32F, 1.0 / 255);

    cv::Mat dst_dct;
    blockDCT(image_gray, dst_dct);

    cv::Mat dst_idct = cv::Mat::zeros(image_gray.size(), image_gray.type());
    blockIDCT(dst_dct, dst_idct, image_gray.size());
    cv::normalize(dst_idct, dst_idct, 0, 1, cv::NORM_MINMAX);

    cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
    cv::imshow("src", image_gray);
    cv::namedWindow("dct", cv::WINDOW_AUTOSIZE);
    cv::imshow("dct", dst_dct);
    cv::namedWindow("idct", cv::WINDOW_AUTOSIZE);
    cv::imshow("idct", dst_idct);
    cv::waitKey(0);
}

void holeFillExample() {
    cv::Mat image_gray = cv::imread(R"(..\image\text-touching-border-half.tif)", 0);

    cv::Mat image_bin;
    grayToBinary(image_gray, image_bin, 127, 255);

    cv::Mat dst;
    holeFill(image_bin, dst);

    cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
    cv::imshow("src", image_bin);
    cv::namedWindow("dst", cv::WINDOW_AUTOSIZE);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}

void borderClearExample() {
    cv::Mat image_gray = cv::imread(R"(..\image\text-touching-border-half.tif)", 0);

    cv::Mat image_bin;
    grayToBinary(image_gray, image_bin, 127, 255);

    cv::Mat dst;
    borderClear(image_bin, dst);

    cv::namedWindow("src", cv::WINDOW_AUTOSIZE);
    cv::imshow("src", image_bin);
    cv::namedWindow("dst", cv::WINDOW_AUTOSIZE);
    cv::imshow("dst", dst);
    cv::waitKey(0);
}

void morphFlattenBackgroundExample() {
    cv::Mat image_gray = cv::imread(R"(..\image\calculator.tif)", 0);

    // 1. 对 原图 进行重建开运算，目的是选出背景（包括上白横线）
    cv::Mat dst1;
    openReconstruct(image_gray, dst1, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(71, 1)));

    // 2. 将 原图 与 dst1 进行相减，目的是去除背景（包括上白横线）
    cv::Mat dst2;
    cv::subtract(image_gray, dst1, dst2);

    // 3. 对 dst2 进行重建开运算，目的是去除右白竖线，但损失了目标前景的竖线
    cv::Mat dst3;
    openReconstruct(dst2, dst3, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11, 1)));

    // 4. 对 dst3 进行膨胀，目的是用于恢复目标前景的竖线
    cv::Mat dst4;
    morphologyDilate(dst3, dst4, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 1)));

    // 5. 对 dst2 与 dst4 进行最小化运算，目的是用于恢复目标前景的竖线
    cv::Mat dst5;
    cv::min(dst2, dst4, dst5);

    // 6. 以 dst2 为模板对 dst5 进行膨胀重建，目的是恢复目标前景的竖线
    cv::Mat dst6;
    dilateReconstruct(dst5, dst2, dst6);

    cv::namedWindow("src", cv::WINDOW_NORMAL);
    cv::imshow("src", image_gray);
    cv::namedWindow("dst", cv::WINDOW_NORMAL);
    cv::imshow("dst", dst6);
    cv::waitKey(0);
}

void globalThresholdEdgeOptExample() {
    // ------------ septagon-small.tif ------------
    cv::Mat image1 = cv::imread(R"(../image/septagon-small.tif)", 0);

    int thresh1 = calcGlobalThresholdEdgeOpt(image1, 1, 0.997, 1);
    cv::threshold(image1, image1, thresh1, 255, cv::THRESH_BINARY);

    cv::namedWindow("image1", cv::WINDOW_AUTOSIZE);
    cv::imshow("image1", image1);
    cv::waitKey(0);

    // ------------ yeast-cells.tif 1 ------------
    cv::Mat image21 = cv::imread(R"(../image/yeast-cells.tif)", 0);

    int thresh21 = calcGlobalThresholdEdgeOpt(image21, 2, 0.995, 2);
    cv::threshold(image21, image21, thresh21, 255, cv::THRESH_BINARY);

    cv::namedWindow("image21", cv::WINDOW_AUTOSIZE);
    cv::imshow("image21", image21);
    cv::waitKey(0);

    // ------------ yeast-cells.tif 2 ------------
    cv::Mat image22 = cv::imread(R"(../image/yeast-cells.tif)", 0);

    int thresh22 = calcGlobalThresholdEdgeOpt(image22, 2, 0.519, 2);
    cv::threshold(image22, image22, thresh22, 255, cv::THRESH_BINARY);

    cv::namedWindow("image22", cv::WINDOW_AUTOSIZE);
    cv::imshow("image22", image22);
    cv::waitKey(0);
}
