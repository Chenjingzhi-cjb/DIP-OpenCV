#include "image_noise.h"


void addNoiseGauss(Mat &src, Mat &dst, int mean, int sigma) {
    if (src.empty()) {
        throw invalid_argument("addNoiseGauss(): Input image is empty!");
    }

    Mat src_copy;
    src.convertTo(src_copy, CV_64FC(src.channels()));

    // 生成高斯噪声图像
    RNG rng;
    Mat noise = Mat::zeros(src.rows, src.cols, CV_64FC(src.channels()));
    rng.fill(noise, RNG::NORMAL, mean, sigma);

    // add
    Mat result;
    add(src_copy, noise, result);
    result.convertTo(result, src.type());

    result.copyTo(dst);
}

void addNoiseMean(Mat &src, Mat &dst, int lower, int upper) {
    if (src.empty()) {
        throw invalid_argument("addNoiseMean(): Input image is empty!");
    }

    Mat src_copy;
    src.convertTo(src_copy, CV_64FC(src.channels()));

    // 生成平均噪声图像
    RNG rng;
    Mat noise = Mat::zeros(src.rows, src.cols, CV_64FC(src.channels()));
    rng.fill(noise, RNG::UNIFORM, lower, upper);

    // add
    Mat result;
    add(src_copy, noise, result);
    result.convertTo(result, src.type());

    result.copyTo(dst);
}

void addNoiseRayleigh(Mat &src, Mat &dst, double sigma) {
    if (src.empty()) {
        throw invalid_argument("addNoiseRayleigh(): Input image is empty!");
    }

    Mat src_copy;
    src.convertTo(src_copy, CV_64FC(src.channels()));

    // Rayleigh: n = sigma * sqrt(-2.0 * log(1.0 - u))
    RNG rng(getTickCount());
    Mat noise = Mat::zeros(src.rows, src.cols, CV_64FC1);
    for (int i = 0; i < noise.rows; i++) {
        for (int j = 0; j < noise.cols; j++) {
            double u = rng.uniform(0.0, 1.0);
            noise.at<double>(i, j) = sigma * sqrt(-2.0 * log(1.0 - u));
        }
    }

    if (src.type() == CV_8UC1 || src.type() == CV_32FC1) {
        // add
        Mat result;
        add(src_copy, noise, result);
        result.convertTo(result, src.type());

        result.copyTo(dst);
    } else if (src.type() == CV_8UC3 || src.type() == CV_32FC3) {
        // three channels
        Mat noises(src.rows, src.cols, CV_64FC3);
        Mat channels[3] = {noise, noise, noise};
        merge(channels, 3, noises);

        // add
        Mat result;
        add(src_copy, noises, result);
        result.convertTo(result, src.type());

        result.copyTo(dst);
    } else {  // other
        cout << "addNoiseRayleigh(): The type of the input image does not meet the requirements. "
             << "Please convert the type before using this function again."
             << endl;
    }
}

void addNoiseGamma(Mat &src, Mat &dst, double sigma, double alpha, double beta) {
    if (src.empty()) {
        throw invalid_argument("addNoiseGamma(): Input image is empty!");
    }

    Mat src_copy;
    src.convertTo(src_copy, CV_64FC(src.channels()));

    // Gamma: n = sigma * pow((-1.0 / alpha) * log(1.0 - u), 1.0 / beta)
    RNG rng(getTickCount());
    Mat noise = Mat::zeros(src.rows, src.cols, CV_64FC1);
    for (int i = 0; i < noise.rows; i++) {
        for (int j = 0; j < noise.cols; j++) {
            double u = rng.uniform(0.0, 1.0);
            noise.at<double>(i, j) = sigma * pow((-1.0 / alpha) * log(1.0 - u), 1.0 / beta);
        }
    }

    if (src.type() == CV_8UC1 || src.type() == CV_32FC1) {
        // add
        Mat result;
        add(src_copy, noise, result);
        result.convertTo(result, src.type());

        result.copyTo(dst);
    } else if (src.type() == CV_8UC3 || src.type() == CV_32FC3) {
        // three channels
        Mat noises(src.rows, src.cols, CV_64FC3);
        Mat channels[3] = {noise, noise, noise};
        merge(channels, 3, noises);

        // add
        Mat result;
        add(src_copy, noises, result);
        result.convertTo(result, src.type());

        result.copyTo(dst);
    } else {  // other
        cout << "addNoiseGamma(): The type of the input image does not meet the requirements. "
             << "Please convert the type before using this function again."
             << endl;
    }
}

void addNoiseExp(Mat &src, Mat &dst, double lambda) {
    if (src.empty()) {
        throw invalid_argument("addNoiseExp(): Input image is empty!");
    }

    Mat src_copy;
    src.convertTo(src_copy, CV_64FC(src.channels()));

    // Exponential: n = -log(1.0 - u) / lambda
    RNG rng(getTickCount());
    Mat noise = Mat::zeros(src.rows, src.cols, CV_64FC1);
    for (int i = 0; i < noise.rows; i++) {
        for (int j = 0; j < noise.cols; j++) {
            double u = rng.uniform(0.0, 1.0);
            noise.at<double>(i, j) = -log(1.0 - u) / lambda;
        }
    }

    if (src.type() == CV_8UC1 || src.type() == CV_32FC1) {
        // add
        Mat result;
        add(src_copy, noise, result);
        result.convertTo(result, src.type());

        result.copyTo(dst);
    } else if (src.type() == CV_8UC3 || src.type() == CV_32FC3) {
        // three channels
        Mat noises(src.rows, src.cols, CV_64FC3);
        Mat channels[3] = {noise, noise, noise};
        merge(channels, 3, noises);

        // add
        Mat result;
        add(src_copy, noises, result);
        result.convertTo(result, src.type());

        result.copyTo(dst);
    } else {  // other
        cout << "addNoiseExp(): The type of the input image does not meet the requirements. "
             << "Please convert the type before using this function again."
             << endl;
    }
}

void addNoiseSaltPepper(Mat &src, Mat &dst, double noise_level, int type, double salt_value, double pepper_value) {
    if (src.empty()) {
        throw invalid_argument("addNoiseSaltPepper(): Input image is empty!");
    }

    if (noise_level <= 0 || noise_level >= 1) {
        string err = R"(addNoiseSaltPepper(): Parameter Error! You should make sure "0 < noise_level < 1"!)";
        throw invalid_argument(err);
    }

    Mat src_copy;
    src.convertTo(src_copy, CV_64FC(src.channels()));

    RNG rng(getTickCount());

    if (src_copy.channels() == 1) {
        for (int i = 0; i < src_copy.rows; i++) {
            for (int j = 0; j < src_copy.cols; j++) {
                double u = rng.uniform(0.0, 1.0);

                if (type == 0) {
                    if (u < noise_level / 2.0) {
                        src_copy.at<double>(i, j) = salt_value;  // Add salt noise, white
                    } else if (u < noise_level) {
                        src_copy.at<double>(i, j) = pepper_value;  // Add pepper noise, black
                    }
                } else if (type == 1) {
                    if (u < noise_level) {
                        src_copy.at<double>(i, j) = salt_value;
                    }
                } else if (type == 2) {
                    if (u < noise_level) {
                        src_copy.at<double>(i, j) = pepper_value;
                    }
                } else {
                    cout << "addNoiseSaltPepper(): The type does not meet the requirements. " << endl;
                    return;
                }
            }
        }
        src_copy.convertTo(src_copy, src.type());

        src_copy.copyTo(dst);
    } else if (src_copy.channels() == 3) {
        for (int i = 0; i < src_copy.rows; i++) {
            for (int j = 0; j < src_copy.cols; j++) {
                double u = rng.uniform(0.0, 1.0);

                if (type == 0) {
                    if (u < noise_level / 2.0) {
                        src_copy.at<Vec3d>(i, j) = Vec3d(salt_value, salt_value, salt_value);
                    } else if (u < noise_level) {
                        src_copy.at<Vec3d>(i, j) = Vec3d(pepper_value, pepper_value, pepper_value);
                    }
                } else if (type == 1) {
                    if (u < noise_level) {
                        src_copy.at<Vec3d>(i, j) = Vec3d(salt_value, salt_value, salt_value);
                    }
                } else if (type == 2) {
                    if (u < noise_level) {
                        src_copy.at<Vec3d>(i, j) = Vec3d(pepper_value, pepper_value, pepper_value);
                    }
                } else {
                    cout << "addNoiseSaltPepper(): The type does not meet the requirements. " << endl;
                    return;
                }
            }
        }
        src_copy.convertTo(src_copy, src.type());

        src_copy.copyTo(dst);
    } else {  // other
        cout << "addNoiseSaltPepper(): The channels of the input image does not meet the requirements. "
             << "Please convert the type before using this function again."
             << endl;
    }
}
