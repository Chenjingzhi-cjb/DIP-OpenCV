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