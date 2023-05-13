#include "color_process.h"


vector<Mat> colorChannelSpilt(Mat &src) {
    if (src.empty()) {
        throw invalid_argument("colorChannelSpilt(): Input image is empty!");
    }

    vector<Mat> planes;
    split(src, planes);

    return move(planes);
}

void bgrToHsi(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("bgrToHsi(): Input image is empty!");
    }

    Mat temp(src.size(), src.type());

    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            Vec3b m = src.at<Vec3b>(r, c);
            double B = (double) m[0] / 255;
            double G = (double) m[1] / 255;
            double R = (double) m[2] / 255;

            double H, S, I;

            double den = sqrt((R - G) * (R - G) + (R - B) * (G - B));
            if (den == 0) {
                H = 0;
            } else {  // den != 0
                double theta = acos((R - G + R - B) / (2 * den));
                H = (B <= G) ? (theta / (2 * M_PI)) : (1 - theta / (2 * M_PI));
            }

            double sum = B + G + R;
            if (sum == 0) {
                S = 0;
            } else {  // sum != 0
                S = 1 - 3 * min(min(B, G), R) / sum;
            }

            I = sum / 3.0;

            temp.at<Vec3b>(r, c) = Vec3b((uchar) (H * 255), (uchar) (S * 255), (uchar) (I * 255));
        }
    }

    temp.copyTo(dst);
}

void hsiToBgr(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("hsiToBgr(): Input image is empty!");
    }

    Mat temp(src.size(), src.type());

    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            Vec3b m = src.at<Vec3b>(r, c);

            double H = (double) m[0] / 255 * 2 * M_PI;
            double S = (double) m[1] / 255;
            double I = (double) m[2] / 255;

            double B, G, R;

            if (H < 120 * M_PI / 180) {
                B = I * (1 - S);
                R = I * (1 + S * cos(H) / cos(60 * M_PI / 180 - H));
                G = 3 * I - R - B;
            } else if (H >= 120 * M_PI / 180 && H < 240 * M_PI / 180) {
                H -= (120 * M_PI / 180);
                R = I * (1 - S);
                G = I * (1 + S * cos(H) / cos(60 * M_PI / 180 - H));
                B = 3 * I - R - G;
            } else if (H >= 240 * M_PI / 180) {
                H -= (240 * M_PI / 180);
                G = I * (1 - S);
                B = I * (1 + S * cos(H) / cos(60 * M_PI / 180 - H));
                R = 3 * I - B - G;
            }

            temp.at<Vec3b>(r, c) = Vec3b((uchar) (B * 255), (uchar) (G * 255), (uchar) (R * 255));
        }
    }

    temp.copyTo(dst);
}

void pseudoColor(Mat &src, Mat &dst, ColormapTypes color) {
    if (src.empty()) {
        throw invalid_argument("pseudoColor(): Input image is empty!");
    }

    applyColorMap(src, dst, color);
}

void complementaryColor(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("complementaryColor(): Input image is empty!");
    }

    src.convertTo(dst, CV_8U, -1, 255);
}

void colorLayering(Mat &src, Mat &dst, const Vec3b &color_bgr, double range_r) {
    if (src.empty()) {
        throw invalid_argument("colorLayering(): Input image is empty!");
    }

    Mat temp(src.size(), src.type());

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            Vec3b m = src.at<Vec3b>(i, j);

            double dis = pow(color_bgr[0] - m[0], 2) + pow(color_bgr[1] - m[1], 2) + pow(color_bgr[2] - m[2], 2);
            if (dis > pow(range_r, 2)) {
                temp.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            } else {  // dis <= pow(range_r, 2)
                temp.at<Vec3b>(i, j) = m;
            }
        }
    }

    temp.copyTo(dst);
}

void colorEqualizeHist(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("colorEqualizeHist(): Input image is empty!");
    }

    vector<Mat> planes = colorChannelSpilt(src);

    for (auto &plane: planes) {
        equalizeHist(plane, plane);
    }

    merge(planes, dst);
}

