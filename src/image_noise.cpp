#include "image_noise.h"


void addNoiseGauss(const cv::Mat &src, cv::Mat &dst, int mean, int sigma) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    cv::Mat src_copy;
    src.convertTo(src_copy, CV_64FC(src.channels()));

    // 生成高斯噪声图像
    cv::RNG rng;
    cv::Mat noise = cv::Mat::zeros(src.rows, src.cols, CV_64FC(src.channels()));
    rng.fill(noise, cv::RNG::NORMAL, mean, sigma);

    // add
    cv::Mat temp;
    cv::add(src_copy, noise, temp);
    temp.convertTo(temp, src.type());

    temp.copyTo(dst);
}

void addNoiseMean(const cv::Mat &src, cv::Mat &dst, int lower, int upper) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    cv::Mat src_copy;
    src.convertTo(src_copy, CV_64FC(src.channels()));

    // 生成平均噪声图像
    cv::RNG rng;
    cv::Mat noise = cv::Mat::zeros(src.rows, src.cols, CV_64FC(src.channels()));
    rng.fill(noise, cv::RNG::UNIFORM, lower, upper);

    // add
    cv::Mat temp;
    cv::add(src_copy, noise, temp);
    temp.convertTo(temp, src.type());

    temp.copyTo(dst);
}

void addNoiseRayleigh(const cv::Mat &src, cv::Mat &dst, double sigma) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    int src_type = src.type();
    if (src_type != CV_8UC1 && src_type != CV_32FC1 &&
        src_type != CV_8UC3 && src_type != CV_32FC3) {
        THROW_ARG_ERROR("Input src image type is not supported. "
                        "Supported types: CV_8UC1, CV_32FC1, CV_8UC3, CV_32FC3.");
    }

    cv::Mat src_copy;
    src.convertTo(src_copy, CV_64FC(src.channels()));

    // Rayleigh: n = sigma * sqrt(-2.0 * log(1.0 - u))
    cv::RNG rng(cv::getTickCount());
    cv::Mat noise = cv::Mat::zeros(src.rows, src.cols, CV_64FC1);
    for (int i = 0; i < noise.rows; i++) {
        for (int j = 0; j < noise.cols; j++) {
            double u = rng.uniform(0.0, 1.0);
            noise.at<double>(i, j) = sigma * std::sqrt(-2.0 * std::log(1.0 - u));
        }
    }

    cv::Mat temp;
    if (src_type == CV_8UC1 || src_type == CV_32FC1) {
        // add
        cv::add(src_copy, noise, temp);
        temp.convertTo(temp, src_type);
    } else {  // CV_8UC3 or CV_32FC3
        // three channels
        cv::Mat noises(src.rows, src.cols, CV_64FC3);
        cv::Mat channels[3] = {noise, noise, noise};
        cv::merge(channels, 3, noises);

        // add
        cv::add(src_copy, noises, temp);
        temp.convertTo(temp, src_type);
    }

    temp.copyTo(dst);
}

void addNoiseGamma(const cv::Mat &src, cv::Mat &dst, double sigma, double alpha, double beta) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    int src_type = src.type();
    if (src_type != CV_8UC1 && src_type != CV_32FC1 &&
        src_type != CV_8UC3 && src_type != CV_32FC3) {
        THROW_ARG_ERROR("Input src image type is not supported. "
                        "Supported types: CV_8UC1, CV_32FC1, CV_8UC3, CV_32FC3.");
    }

    cv::Mat src_copy;
    src.convertTo(src_copy, CV_64FC(src.channels()));

    // Gamma: n = sigma * pow((-1.0 / alpha) * log(1.0 - u), 1.0 / beta)
    cv::RNG rng(cv::getTickCount());
    cv::Mat noise = cv::Mat::zeros(src.rows, src.cols, CV_64FC1);
    for (int i = 0; i < noise.rows; i++) {
        for (int j = 0; j < noise.cols; j++) {
            double u = rng.uniform(0.0, 1.0);
            noise.at<double>(i, j) = sigma * pow((-1.0 / alpha) * log(1.0 - u), 1.0 / beta);
        }
    }

    cv::Mat temp;
    if (src_type == CV_8UC1 || src_type == CV_32FC1) {
        // add
        cv::add(src_copy, noise, temp);
        temp.convertTo(temp, src_type);
    } else {  // CV_8UC3 or CV_32FC3
        // three channels
        cv::Mat noises(src.rows, src.cols, CV_64FC3);
        cv::Mat channels[3] = {noise, noise, noise};
        cv::merge(channels, 3, noises);

        // add
        cv::add(src_copy, noises, temp);
        temp.convertTo(temp, src_type);
    }

    temp.copyTo(dst);
}

void addNoiseExp(const cv::Mat &src, cv::Mat &dst, double lambda) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    int src_type = src.type();
    if (src_type != CV_8UC1 && src_type != CV_32FC1 &&
        src_type != CV_8UC3 && src_type != CV_32FC3) {
        THROW_ARG_ERROR("Input src image type is not supported. "
                        "Supported types: CV_8UC1, CV_32FC1, CV_8UC3, CV_32FC3.");
    }

    cv::Mat src_copy;
    src.convertTo(src_copy, CV_64FC(src.channels()));

    // Exponential: n = -log(1.0 - u) / lambda
    cv::RNG rng(cv::getTickCount());
    cv::Mat noise = cv::Mat::zeros(src.rows, src.cols, CV_64FC1);
    for (int i = 0; i < noise.rows; i++) {
        for (int j = 0; j < noise.cols; j++) {
            double u = rng.uniform(0.0, 1.0);
            noise.at<double>(i, j) = -log(1.0 - u) / lambda;
        }
    }

    cv::Mat temp;
    if (src_type == CV_8UC1 || src_type == CV_32FC1) {
        // add
        cv::add(src_copy, noise, temp);
        temp.convertTo(temp, src_type);
    } else {  // CV_8UC3 or CV_32FC3
        // three channels
        cv::Mat noises(src.rows, src.cols, CV_64FC3);
        cv::Mat channels[3] = {noise, noise, noise};
        cv::merge(channels, 3, noises);

        // add
        cv::add(src_copy, noises, temp);
        temp.convertTo(temp, src_type);
    }

    temp.copyTo(dst);
}

void addNoiseSaltPepper(const cv::Mat &src, cv::Mat &dst, double noise_level, int type,
                        double salt_value, double pepper_value) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    if (src.channels() != 1 && src.channels() != 3) {
        THROW_ARG_ERROR("Input src image channels is not supported: {}", src.channels());
    }

    if (noise_level <= 0 || noise_level >= 1) {
        THROW_ARG_ERROR("Invalid `noise_level`: {}. You should make sure `0 < noise_level < 1`.", noise_level);
    }

    if (type < 0 || type > 2) {
        THROW_ARG_ERROR("Invalid `type`: {}.", type);
    }

    cv::Mat temp;
    src.convertTo(temp, CV_64FC(src.channels()));

    cv::RNG rng(cv::getTickCount());

    if (temp.channels() == 1) {
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.cols; j++) {
                double u = rng.uniform(0.0, 1.0);

                if (type == 0) {
                    if (u < noise_level / 2.0) {
                        temp.at<double>(i, j) = salt_value;  // Add salt noise, white
                    } else if (u < noise_level) {
                        temp.at<double>(i, j) = pepper_value;  // Add pepper noise, black
                    }
                } else if (type == 1) {
                    if (u < noise_level) {
                        temp.at<double>(i, j) = salt_value;
                    }
                } else {  // type == 2
                    if (u < noise_level) {
                        temp.at<double>(i, j) = pepper_value;
                    }
                }
            }
        }
    } else {  // temp.channels() == 3
        for (int i = 0; i < temp.rows; i++) {
            for (int j = 0; j < temp.cols; j++) {
                double u = rng.uniform(0.0, 1.0);

                if (type == 0) {
                    if (u < noise_level / 2.0) {
                        temp.at<cv::Vec3d>(i, j) = cv::Vec3d(salt_value, salt_value, salt_value);
                    } else if (u < noise_level) {
                        temp.at<cv::Vec3d>(i, j) = cv::Vec3d(pepper_value, pepper_value, pepper_value);
                    }
                } else if (type == 1) {
                    if (u < noise_level) {
                        temp.at<cv::Vec3d>(i, j) = cv::Vec3d(salt_value, salt_value, salt_value);
                    }
                } else {  // type == 2
                    if (u < noise_level) {
                        temp.at<cv::Vec3d>(i, j) = cv::Vec3d(pepper_value, pepper_value, pepper_value);
                    }
                }
            }
        }
    }
    temp.convertTo(temp, src.type());

    temp.copyTo(dst);
}
