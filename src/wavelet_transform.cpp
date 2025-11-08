#include "wavelet_transform.h"


// 构建 小波变换 低通、高通 滤波器
void waveletFilterD(const std::string &wname, cv::Mat &low_filter, cv::Mat &high_filter) {
    if (wname == "haar" || wname == "db1") {
        int N = 2;

        low_filter = cv::Mat::zeros(1, N, CV_32F);
        high_filter = cv::Mat::zeros(1, N, CV_32F);

        low_filter.at<float>(0, 0) = 1 / std::sqrt((float) N);
        low_filter.at<float>(0, 1) = 1 / std::sqrt((float) N);

        high_filter.at<float>(0, 0) = -1 / std::sqrt((float) N);
        high_filter.at<float>(0, 1) = 1 / std::sqrt((float) N);
    } else if (wname == "sym2") {
        int N = 4;
        float h[] = {-0.4830, 0.8365, -0.2241, -0.1294};
        float l[] = {-0.1294, 0.2241, 0.8365, 0.4830};

        low_filter = cv::Mat::zeros(1, N, CV_32F);
        high_filter = cv::Mat::zeros(1, N, CV_32F);

        for (int i = 0; i < N; i++) {
            low_filter.at<float>(0, i) = l[i];
            high_filter.at<float>(0, i) = h[i];
        }
    } else {
        THROW_ARG_ERROR("Input `wname` is not supported. "
                        "Supported `wname`: \"haar\", \"db1\", \"sym2\".");
    }
}

// 小波分解
cv::Mat waveletDecompose(const cv::Mat &src, const cv::Mat &low_filter, const cv::Mat &high_filter) {
    if (src.rows != 1 || low_filter.rows != 1 || high_filter.rows != 1) {
        THROW_ARG_ERROR("Input src image and filters must have only 1 row.");
    }
    if (src.cols < low_filter.cols || src.cols < high_filter.cols) {
        THROW_ARG_ERROR("Input src image columns must be >= input filters columns.");
    }

    cv::Mat_<float> temp = cv::Mat_<float>(src).clone();
    cv::Mat_<float> low_filter_mat = cv::Mat_<float>(low_filter);
    cv::Mat_<float> high_filter_mat = cv::Mat_<float>(high_filter);

    int cols = temp.cols;

    // 进行 频域滤波 或 时域卷积: ifft(fft(x) * fft(filter)) = cov(x, filter)
    cv::Mat dst_low = cv::Mat::zeros(1, cols, temp.type());
    cv::Mat dst_high = cv::Mat::zeros(1, cols, temp.type());
    cv::filter2D(temp, dst_low, -1, low_filter_mat);
    cv::filter2D(temp, dst_high, -1, high_filter_mat);

    // 下采样，数据拼接
    for (int i = 0, j = 1; i < cols / 2; i++, j += 2) {
        temp.at<float>(0, i) = dst_low.at<float>(0, j);
        temp.at<float>(0, i + cols / 2) = dst_high.at<float>(0, j);
    }

    return temp;
}

// 构建 小波逆变换 低通、高通 滤波器
void waveletFilterR(const std::string &wname, cv::Mat &low_filter, cv::Mat &high_filter) {
    if (wname == "haar" || wname == "db1") {
        int N = 2;

        low_filter = cv::Mat::zeros(1, N, CV_32F);
        high_filter = cv::Mat::zeros(1, N, CV_32F);

        low_filter.at<float>(0, 0) = 1 / std::sqrt((float) N);
        low_filter.at<float>(0, 1) = 1 / std::sqrt((float) N);

        high_filter.at<float>(0, 0) = 1 / std::sqrt((float) N);
        high_filter.at<float>(0, 1) = -1 / std::sqrt((float) N);
    } else if (wname == "sym2") {
        int N = 4;
        float h[] = {-0.1294, -0.2241, 0.8365, -0.4830};
        float l[] = {0.4830, 0.8365, 0.2241, -0.1294};

        low_filter = cv::Mat::zeros(1, N, CV_32F);
        high_filter = cv::Mat::zeros(1, N, CV_32F);

        for (int i = 0; i < N; i++) {
            low_filter.at<float>(0, i) = l[i];
            high_filter.at<float>(0, i) = h[i];
        }
    } else {
        THROW_ARG_ERROR("Input `wname` is not supported. "
                        "Supported `wname`: \"haar\", \"db1\", \"sym2\".");
    }
}

// 小波重建
cv::Mat waveletReconstruct(const cv::Mat &src, const cv::Mat &low_filter, const cv::Mat &high_filter) {
    if (src.rows != 1 || low_filter.rows != 1 || high_filter.rows != 1) {
        THROW_ARG_ERROR("Input src image and filters must have only 1 row.");
    }
    if (src.cols < low_filter.cols || src.cols < high_filter.cols) {
        THROW_ARG_ERROR("Input src image columns must be >= input filters columns.");
    }

    cv::Mat_<float> temp = cv::Mat_<float>(src).clone();
    cv::Mat_<float> low_filter_mat = cv::Mat_<float>(low_filter);
    cv::Mat_<float> high_filter_mat = cv::Mat_<float>(high_filter);

    int cols = temp.cols;

    // 插值
    cv::Mat dst_low = cv::Mat::zeros(1, cols, temp.type());
    cv::Mat dst_high = cv::Mat::zeros(1, cols, temp.type());
    for (int i = 0, j = 0; i < cols / 2; i++, j += 2) {
        dst_low.at<float>(0, j) = temp.at<float>(0, i);
        dst_high.at<float>(0, j) = temp.at<float>(0, i + cols / 2);
    }

    // 滤波
    cv::filter2D(dst_low, dst_low, -1, low_filter_mat);
    cv::filter2D(dst_high, dst_high, -1, high_filter_mat);

    return (dst_low + dst_high);
}

void DWT(const cv::Mat &src, cv::Mat &dst, const std::string &wname, int level) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.channels() != 1) {
        THROW_ARG_ERROR("The number of channels for the input src image is not supported: {}", src.channels());
    }
    if (wname != "haar" && wname != "db1" && wname != "sym2") {
        THROW_ARG_ERROR("Input `wname` is not supported. "
                        "Supported `wname`: \"haar\", \"db1\", \"sym2\".");
    }

    cv::Mat_<float> temp = cv::Mat_<float>(src).clone();

    // 构建 低通、高通 滤波器
    cv::Mat low_filter;
    cv::Mat high_filter;
    waveletFilterD(wname, low_filter, high_filter);

    // 小波变换
    int rows = src.rows;
    int cols = src.cols;
    for (int t = 1; t <= level; t++) {
        // 行小波变换
        for (int r = 0; r < rows; r++) {
            cv::Mat one_row = cv::Mat::zeros(1, cols, temp.type());
            for (int j = 0; j < cols; j++) {
                one_row.at<float>(0, j) = temp.at<float>(r, j);
            }

            one_row = waveletDecompose(one_row, low_filter, high_filter);
            for (int j = 0; j < cols; j++) {
                temp.at<float>(r, j) = one_row.at<float>(0, j);
            }
        }

        // 列小波变换
        for (int c = 0; c < cols; c++) {
            cv::Mat one_col = cv::Mat::zeros(rows, 1, temp.type());
            for (int i = 0; i < rows; i++) {
                one_col.at<float>(i, 0) = temp.at<float>(i, c);
            }

            one_col = (waveletDecompose(one_col.t(), low_filter, high_filter)).t();
            for (int i = 0; i < rows; i++) {
                temp.at<float>(i, c) = one_col.at<float>(i, 0);
            }
        }

        // 更新
        rows /= 2;
        cols /= 2;
    }

    temp.copyTo(dst);
}

void IDWT(const cv::Mat &src, cv::Mat &dst, const std::string &wname, int level) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (src.channels() != 1) {
        THROW_ARG_ERROR("The number of channels for the input src image is not supported: {}", src.channels());
    }
    if (wname != "haar" && wname != "db1" && wname != "sym2") {
        THROW_ARG_ERROR("Input `wname` is not supported. "
                        "Supported `wname`: \"haar\", \"db1\", \"sym2\".");
    }

    cv::Mat_<float> temp = cv::Mat_<float>(src).clone();

    // 构建 低通、高通 滤波器
    cv::Mat low_filter;
    cv::Mat high_filter;
    waveletFilterR(wname, low_filter, high_filter);

    // 小波逆变换
    int rows = (int) (src.rows / std::pow(2.0, level - 1));
    int cols = (int) (src.cols / std::pow(2.0, level - 1));
    for (int t = 1; t <= level; t++) {
        // 列小波逆变换
        for (int c = 0; c < cols; c++) {
            cv::Mat one_col = cv::Mat::zeros(rows, 1, temp.type());
            for (int i = 0; i < rows; i++) {
                one_col.at<float>(i, 0) = temp.at<float>(i, c);
            }

            one_col = (waveletReconstruct(one_col.t(), low_filter, high_filter)).t();
            for (int i = 0; i < rows; i++) {
                temp.at<float>(i, c) = one_col.at<float>(i, 0);
            }
        }

        // 行小波逆变换
        for (int r = 0; r < rows; r++) {
            cv::Mat one_row = cv::Mat::zeros(1, cols, temp.type());
            for (int j = 0; j < cols; j++) {
                one_row.at<float>(0, j) = temp.at<float>(r, j);
            }

            one_row = waveletReconstruct(one_row, low_filter, high_filter);
            for (int j = 0; j < cols; j++) {
                temp.at<float>(r, j) = one_row.at<float>(0, j);
            }
        }

        // 更新
        rows *= 2;
        cols *= 2;
    }

    temp.copyTo(dst);
}

void DCT(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }

    cv::Mat_<float> temp = cv::Mat_<float>(src).clone();

    // 对输入矩阵进行零填充到合适的尺寸，使用镜像填充
    int M = cv::getOptimalDFTSize(temp.rows);
    int N = cv::getOptimalDFTSize(temp.cols);
    cv::Mat padded;
    cv::copyMakeBorder(temp, padded, 0, M - temp.rows, 0, N - temp.cols, cv::BORDER_REFLECT_101,
                       cv::Scalar::all(0));

    // 离散余弦变换
    cv::dct(padded, temp);

    temp.copyTo(dst);
}

void IDCT(const cv::Mat &src, cv::Mat &dst, const cv::Size &original_size) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (original_size.width <= 0 || original_size.height <= 0) {
        THROW_ARG_ERROR("Invalid `original_size`.");
    }

    cv::Mat_<float> temp = cv::Mat_<float>(src).clone();

    // 离散余弦逆变换
    cv::idct(temp, temp);

    // 裁剪至原尺寸
    temp = temp(cv::Rect(0, 0, original_size.width, original_size.height));

    temp.copyTo(dst);
}

void blockDCT(const cv::Mat &src, cv::Mat &dst, int block_size) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (block_size <= 0) {
        THROW_ARG_ERROR("Invalid `block_size`.");
    }

    cv::Mat_<float> temp = cv::Mat_<float>(src).clone();

    // 对输入矩阵进行零填充到合适的尺寸，使用镜像填充
    int M = cv::getOptimalDFTSize(temp.rows);
    int N = cv::getOptimalDFTSize(temp.cols);
    cv::Mat padded;
    cv::copyMakeBorder(temp, padded, 0, M - temp.rows, 0, N - temp.cols, cv::BORDER_REFLECT_101,
                       cv::Scalar::all(0));

    // 分块处理
    for (int i = 0; i < padded.rows; i += block_size) {
        for (int j = 0; j < padded.cols; j += block_size) {
            // 获取块
            cv::Mat block = padded(cv::Rect(j, i, block_size, block_size));

            // 对块进行 dct
            cv::dct(block, block);
        }
    }

    padded.copyTo(dst);
}

void blockIDCT(const cv::Mat &src, cv::Mat &dst, const cv::Size &original_size, int block_size) {
    if (src.empty()) {
        THROW_ARG_ERROR("Input src image is empty.");
    }
    if (original_size.width <= 0 || original_size.height <= 0) {
        THROW_ARG_ERROR("Invalid `original_size`.");
    }
    if (block_size <= 0) {
        THROW_ARG_ERROR("Invalid `block_size`.");
    }

    cv::Mat_<float> temp = cv::Mat_<float>(src).clone();

    // 分块处理
    for (int i = 0; i < temp.rows; i += block_size) {
        for (int j = 0; j < temp.cols; j += block_size) {
            // 获取块
            cv::Mat block = temp(cv::Rect(j, i, block_size, block_size));

            // 对块执行 idct
            cv::idct(block, block);
        }
    }

    // 裁剪至原尺寸
    temp = temp(cv::Rect(0, 0, original_size.width, original_size.height));

    temp.copyTo(dst);
}
