#include "example.h"


void localEqualizeHistExample() {
    Mat image_gray = imread(R"(..\image\hidden-symbols.tif)", 0);

    Mat dst;
    localEqualizeHist(image_gray, dst);

    namedWindow("src", WINDOW_AUTOSIZE);
    imshow("src", image_gray);
    namedWindow("dst", WINDOW_AUTOSIZE);
    imshow("dst", dst);
    waitKey(0);
}

void shadingCorrectionExample() {
    Mat image_gray = imread(R"(..\image\checkerboard512-shaded.tif)", 0);

    Mat dst;
    shadingCorrection(image_gray, dst);

    namedWindow("src", WINDOW_AUTOSIZE);
    imshow("src", image_gray);
    namedWindow("dst", WINDOW_AUTOSIZE);
    imshow("dst", dst);
    waitKey(0);
}

void domainTransformExample() {
    Mat image_gray = imread(R"(..\image\barbara.tif)", 0);

    Mat dst_complex, dst_magnitude;
    spatialToFrequency(image_gray, dst_complex);
    splitFrequencyMagnitude(dst_complex, dst_magnitude);

    Mat dst_idft = Mat::zeros(image_gray.size(), image_gray.depth());
    frequencyToSpatial(dst_complex, dst_idft);
    normalize(dst_idft, dst_idft, 0, 255, NORM_MINMAX, CV_8U);

    namedWindow("src", WINDOW_AUTOSIZE);
    imshow("src", image_gray);
    namedWindow("dft", WINDOW_AUTOSIZE);
    imshow("dft", dst_magnitude);
    namedWindow("idft", WINDOW_AUTOSIZE);
    imshow("idft", dst_idft);
    waitKey(0);
}

void highFreqEmphasisExample() {
    Mat image_gray = imread(R"(..\image\chestXray.tif)", 0);

    Mat kernel = highFreqEmphasisKernel(image_gray.size(), 70);

    Mat dst;
    frequencyFilter(image_gray, dst, kernel);

    namedWindow("src", WINDOW_AUTOSIZE);
    imshow("src", image_gray);
    namedWindow("dst", WINDOW_AUTOSIZE);
    imshow("dst", dst);
    waitKey(0);
}

void hsiExample() {
    Mat image = imread(R"(..\image\strawberries-RGB.tif)");

    Mat dst;
    bgrToHsi(image, dst);

    namedWindow("src", WINDOW_AUTOSIZE);
    imshow("src", image);
    namedWindow("dst", WINDOW_AUTOSIZE);
    imshow("dst", dst);
    waitKey(0);
}

void DCTExample() {
    Mat image_gray = imread(R"(..\image\barbara.tif)", 0);
    image_gray.convertTo(image_gray, CV_32F, 1.0 / 255);

    Mat dst_dct;
    DCT(image_gray, dst_dct);

    Mat dst_idct = Mat::zeros(image_gray.size(), image_gray.type());
    IDCT(dst_dct, dst_idct);
    normalize(dst_idct, dst_idct, 0, 1, NORM_MINMAX);

    namedWindow("src", WINDOW_AUTOSIZE);
    imshow("src", image_gray);
    namedWindow("dct", WINDOW_AUTOSIZE);
    imshow("dct", dst_dct);
    namedWindow("idct", WINDOW_AUTOSIZE);
    imshow("idct", dst_idct);
    waitKey(0);
}

void blockDCTExample() {
    Mat image_gray = imread(R"(..\image\barbara.tif)", 0);
    image_gray.convertTo(image_gray, CV_32F, 1.0 / 255);

    Mat dst_dct;
    blockDCT(image_gray, dst_dct);

    Mat dst_idct = Mat::zeros(image_gray.size(), image_gray.type());
    blockIDCT(dst_dct, dst_idct);
    normalize(dst_idct, dst_idct, 0, 1, NORM_MINMAX);

    namedWindow("src", WINDOW_AUTOSIZE);
    imshow("src", image_gray);
    namedWindow("dct", WINDOW_AUTOSIZE);
    imshow("dct", dst_dct);
    namedWindow("idct", WINDOW_AUTOSIZE);
    imshow("idct", dst_idct);
    waitKey(0);
}

void holeFillExample() {
    Mat image_gray = imread(R"(..\image\text-touching-border-half.tif)", 0);

    Mat image_bin;
    grayToBinary(image_gray, image_bin, 127, 255);

    Mat dst;
    holeFill(image_bin, dst);

    namedWindow("src", WINDOW_AUTOSIZE);
    imshow("src", image_bin);
    namedWindow("dst", WINDOW_AUTOSIZE);
    imshow("dst", dst);
    waitKey(0);
}

void borderClearExample() {
    Mat image_gray = imread(R"(..\image\text-touching-border-half.tif)", 0);

    Mat image_bin;
    grayToBinary(image_gray, image_bin, 127, 255);

    Mat dst;
    borderClear(image_bin, dst);

    namedWindow("src", WINDOW_AUTOSIZE);
    imshow("src", image_bin);
    namedWindow("dst", WINDOW_AUTOSIZE);
    imshow("dst", dst);
    waitKey(0);
}

void morphFlattenBackgroundExample() {
    Mat image_gray = imread(R"(..\image\calculator.tif)", 0);

    // 1. 对 原图 进行重建开运算，目的是选出背景（包括上白横线）
    Mat dst1;
    openReconstruct(image_gray, dst1, getStructuringElement(MORPH_RECT, Size(71, 1)));

    // 2. 将 原图 与 dst1 进行相减，目的是去除背景（包括上白横线）
    Mat dst2;
    subtract(image_gray, dst1, dst2);

    // 3. 对 dst2 进行重建开运算，目的是去除右白竖线，但损失了目标前景的竖线
    Mat dst3;
    openReconstruct(dst2, dst3, getStructuringElement(MORPH_RECT, Size(11, 1)));

    // 4. 对 dst3 进行膨胀，目的是用于恢复目标前景的竖线
    Mat dst4;
    morphologyDilate(dst3, dst4, getStructuringElement(MORPH_RECT, Size(21, 1)));

    // 5. 对 dst2 与 dst4 进行最小化运算，目的是用于恢复目标前景的竖线
    Mat dst5;
    cv::min(dst2, dst4, dst5);

    // 6. 以 dst2 为模板对 dst5 进行膨胀重建，目的是恢复目标前景的竖线
    Mat dst6;
    dilateReconstruct(dst5, dst2, dst6);

    namedWindow("src", WINDOW_NORMAL);
    imshow("src", image_gray);
    namedWindow("dst", WINDOW_NORMAL);
    imshow("dst", dst6);
    waitKey(0);
}

void globalThresholdEdgeOptExample() {
    // ------------ septagon-small.tif ------------
    Mat image1 = imread(R"(../image/septagon-small.tif)", 0);

    threshold(image1, image1, calcGlobalThresholdEdgeOpt(image1, 1, 0.997, 1), 255, THRESH_BINARY);

    namedWindow("image1", WINDOW_AUTOSIZE);
    imshow("image1", image1);
    waitKey(0);

    // ------------ yeast-cells.tif ------------
    Mat image21 = imread(R"(../image/yeast-cells.tif)", 0);

    threshold(image21, image21, calcGlobalThresholdEdgeOpt(image21, 2, 0.995, 2), 255, THRESH_BINARY);

    namedWindow("image21", WINDOW_AUTOSIZE);
    imshow("image21", image21);
    waitKey(0);

    Mat image22 = imread(R"(../image/yeast-cells.tif)", 0);

    threshold(image22, image22, calcGlobalThresholdEdgeOpt(image22, 2, 0.519, 2), 255, THRESH_BINARY);

    namedWindow("image22", WINDOW_AUTOSIZE);
    imshow("image22", image22);
    waitKey(0);
}
