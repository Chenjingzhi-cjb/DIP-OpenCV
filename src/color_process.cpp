#include "color_process.h"


std::vector<cv::Mat> colorChannelSpilt(const cv::Mat &src) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    std::vector<cv::Mat> planes;
    cv::split(src, planes);

    return planes;
}

cv::Mat colorChannelMerge(const std::vector<cv::Mat> &channels) {
    if (channels.empty()) {
        THROW_ARG_ERROR("Input channels is empty.");
    }

    cv::Mat temp;
    cv::merge(channels, temp);

    return temp;
}

void bgrToHsi(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC3) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC3.");
    }

    cv::Mat temp(src.size(), src.type());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            cv::Vec3b m = src.at<cv::Vec3b>(r, c);
            double B = (double) m[0] / 255;
            double G = (double) m[1] / 255;
            double R = (double) m[2] / 255;

            double H, S, I;

            double den = std::sqrt((R - G) * (R - G) + (R - B) * (G - B));
            if (den == 0) {
                H = 0;
            } else {
                double theta = std::acos((R - G + R - B) / (2 * den));
                H = (B <= G) ? (theta / (2 * CV_PI)) : (1 - theta / (2 * CV_PI));
            }

            double sum = B + G + R;
            if (sum == 0) {
                S = 0;
            } else {
                S = 1 - 3 * std::min(std::min(B, G), R) / sum;
            }

            I = sum / 3.0;

            temp.at<cv::Vec3b>(r, c) = cv::Vec3b((uchar) (H * 255), (uchar) (S * 255), (uchar) (I * 255));
        }
    }

    temp.copyTo(dst);
}

void hsiToBgr(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC3) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC3.");
    }

    cv::Mat temp(src.size(), src.type());
    for (int r = 0; r < src.rows; ++r) {
        for (int c = 0; c < src.cols; ++c) {
            cv::Vec3b m = src.at<cv::Vec3b>(r, c);

            double H = (double) m[0] / 255 * 2 * CV_PI;
            double S = (double) m[1] / 255;
            double I = (double) m[2] / 255;

            double B, G, R;

            if (H < 120 * CV_PI / 180) {
                B = I * (1 - S);
                R = I * (1 + S * std::cos(H) / std::cos(60 * CV_PI / 180 - H));
                G = 3 * I - R - B;
            } else if (H >= 120 * CV_PI / 180 && H < 240 * CV_PI / 180) {
                H -= (120 * CV_PI / 180);
                R = I * (1 - S);
                G = I * (1 + S * std::cos(H) / std::cos(60 * CV_PI / 180 - H));
                B = 3 * I - R - G;
            } else {  // H >= 240 * CV_PI / 180
                H -= (240 * CV_PI / 180);
                G = I * (1 - S);
                B = I * (1 + S * std::cos(H) / std::cos(60 * CV_PI / 180 - H));
                R = 3 * I - B - G;
            }

            temp.at<cv::Vec3b>(r, c) = cv::Vec3b((uchar) (B * 255), (uchar) (G * 255), (uchar) (R * 255));
        }
    }

    temp.copyTo(dst);
}

void pseudoColor(const cv::Mat &src, cv::Mat &dst, cv::ColormapTypes color) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    cv::applyColorMap(src, dst, color);
}

void complementaryColor(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    src.convertTo(dst, CV_8U, -1, 255);
}

void colorLayering(const cv::Mat &src, cv::Mat &dst, const cv::Vec3b &color_center, double range_radius) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC3) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC3.");
    }
    if (range_radius <= 0) {
        THROW_ARG_ERROR("Invalid `range_radius`: {}. You should make sure `range_radius > 0`.", range_radius);
    }

    double radius_squared = range_radius * range_radius;

    cv::Mat temp(src.size(), src.type());
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            cv::Vec3b m = src.at<cv::Vec3b>(i, j);

            double dis = std::pow(color_center[0] - m[0], 2) +
                         std::pow(color_center[1] - m[1], 2) +
                         std::pow(color_center[2] - m[2], 2);

            if (dis > radius_squared) {
                temp.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
            } else {
                temp.at<cv::Vec3b>(i, j) = m;
            }
        }
    }

    temp.copyTo(dst);
}

void colorEqualizeHist(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.type() != CV_8UC3) {
        THROW_ARG_ERROR("Input src image type must be CV_8UC3.");
    }

    std::vector<cv::Mat> planes = colorChannelSpilt(src);

    for (auto &plane: planes) {
        cv::equalizeHist(plane, plane);
    }

    cv::merge(planes, dst);
}
