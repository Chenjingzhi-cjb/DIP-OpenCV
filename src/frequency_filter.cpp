#include "frequency_filter.h"


void spatialToFrequency(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("spatialToFrequency(): Input image is empty!");
    }

    // 对输入矩阵进行零填充到合适的尺寸
    int M = getOptimalDFTSize(src.rows);
    int N = getOptimalDFTSize(src.cols);
    Mat padded;
    copyMakeBorder(src, padded, 0, M - src.rows, 0, N - src.cols, BORDER_CONSTANT,
                   Scalar::all(0));

    // 将输入矩阵拓展为复数，实部虚部均为浮点型
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complex_image;
    merge(planes, 2, complex_image);

    // 进行离散傅里叶变换
    dft(complex_image, complex_image);

    // 从复数输出矩阵中分离出实部（即幅值），并进行对数变换，即 compute log(1 + sqrt(Re(DFT(src))**2 + Im(DFT(src))**2))
    split(complex_image, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magnitude_image = planes[0];
    magnitude_image += Scalar::all(1);
    log(magnitude_image, magnitude_image);

    // 将频谱的行数和列数裁剪到偶数
    magnitude_image = magnitude_image(Rect(0, 0, magnitude_image.cols & -2, magnitude_image.rows & -2));

    // 重新排列傅里叶图像的象限，使原点位于图像中心
    int cx = magnitude_image.cols / 2;
    int cy = magnitude_image.rows / 2;
    Mat tmp;
    Mat q0(magnitude_image, Rect(0, 0, cx, cy));
    Mat q1(magnitude_image, Rect(cx, 0, cx, cy));
    Mat q2(magnitude_image, Rect(0, cy, cx, cy));
    Mat q3(magnitude_image, Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    // 对幅值进行归一化操作，因为 type 为 float，所以用 imshow() 时会将像素值乘以 255
    normalize(magnitude_image, magnitude_image, 0, 1, NORM_MINMAX);

    magnitude_image.copyTo(dst);
}

void frequencyToSpatial(Mat &src, Mat &dst, int flags, int nonzeroRows) {
    if (src.empty()) {
        throw invalid_argument("frequencyToSpatial(): Input image is empty!");
    }

    Mat temp;

    idft(src, temp, flags, nonzeroRows);

    temp.copyTo(dst);
}

