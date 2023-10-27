#include "wavelet_transform.h"


// 构建 小波变换 低通、高通 滤波器
void waveletFilterD(const string &_wname, Mat &_lowFilter, Mat &_highFilter) {
    if (_wname == "haar" || _wname == "db1") {
        int N = 2;

        _lowFilter = Mat::zeros(1, N, CV_32F);
        _highFilter = Mat::zeros(1, N, CV_32F);

        _lowFilter.at<float>(0, 0) = 1 / sqrtf((float) N);
        _lowFilter.at<float>(0, 1) = 1 / sqrtf((float) N);

        _highFilter.at<float>(0, 0) = -1 / sqrtf((float) N);
        _highFilter.at<float>(0, 1) = 1 / sqrtf((float) N);
    } else if (_wname == "sym2") {
        int N = 4;
        float h[] = {-0.4830, 0.8365, -0.2241, -0.1294};
        float l[] = {-0.1294, 0.2241, 0.8365, 0.4830};

        _lowFilter = Mat::zeros(1, N, CV_32F);
        _highFilter = Mat::zeros(1, N, CV_32F);

        for (int i = 0; i < N; i++) {
            _lowFilter.at<float>(0, i) = l[i];
            _highFilter.at<float>(0, i) = h[i];
        }
    }
}

// 小波分解
Mat waveletDecompose(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter) {
    assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
    assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);

    Mat_<float> temp = Mat_<float>(_src).clone();
    Mat_<float> lowFilter = Mat_<float>(_lowFilter);
    Mat_<float> highFilter = Mat_<float>(_highFilter);

    int cols = temp.cols;

    // 进行 频域滤波 或 时域卷积: ifft(fft(x) * fft(filter)) = cov(x, filter)
    Mat dst_low = Mat::zeros(1, cols, temp.type());
    Mat dst_high = Mat::zeros(1, cols, temp.type());
    filter2D(temp, dst_low, -1, lowFilter);
    filter2D(temp, dst_high, -1, highFilter);

    // 下采样，数据拼接
    for (int i = 0, j = 1; i < cols / 2; i++, j += 2) {
        temp.at<float>(0, i) = dst_low.at<float>(0, j);
        temp.at<float>(0, i + cols / 2) = dst_high.at<float>(0, j);
    }

    return temp;
}

// 构建 小波逆变换 低通、高通 滤波器
void waveletFilterR(const string &_wname, Mat &_lowFilter, Mat &_highFilter) {
    if (_wname == "haar" || _wname == "db1") {
        int N = 2;

        _lowFilter = Mat::zeros(1, N, CV_32F);
        _highFilter = Mat::zeros(1, N, CV_32F);

        _lowFilter.at<float>(0, 0) = 1 / sqrtf((float) N);
        _lowFilter.at<float>(0, 1) = 1 / sqrtf((float) N);

        _highFilter.at<float>(0, 0) = 1 / sqrtf((float) N);
        _highFilter.at<float>(0, 1) = -1 / sqrtf((float) N);
    } else if (_wname == "sym2") {
        int N = 4;
        float h[] = {-0.1294, -0.2241, 0.8365, -0.4830};
        float l[] = {0.4830, 0.8365, 0.2241, -0.1294};

        _lowFilter = Mat::zeros(1, N, CV_32F);
        _highFilter = Mat::zeros(1, N, CV_32F);

        for (int i = 0; i < N; i++) {
            _lowFilter.at<float>(0, i) = l[i];
            _highFilter.at<float>(0, i) = h[i];
        }
    }
}

// 小波重建
Mat waveletReconstruct(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter) {
    assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
    assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);

    Mat_<float> temp = Mat_<float>(_src).clone();
    Mat_<float> lowFilter = Mat_<float>(_lowFilter);
    Mat_<float> highFilter = Mat_<float>(_highFilter);

    int cols = temp.cols;

    // 插值
    Mat dst_low = Mat::zeros(1, cols, temp.type());
    Mat dst_high = Mat::zeros(1, cols, temp.type());
    for (int i = 0, j = 0; i < cols / 2; i++, j += 2) {
        dst_low.at<float>(0, j) = temp.at<float>(0, i);
        dst_high.at<float>(0, j) = temp.at<float>(0, i + cols / 2);
    }

    // 滤波
    filter2D(dst_low, dst_low, -1, lowFilter);
    filter2D(dst_high, dst_high, -1, highFilter);

    return (dst_low + dst_high);
}

void DWT(Mat &src, Mat &dst, const string &wname, int level) {
    if (src.empty()) {
        throw invalid_argument("DWT(): Input image is empty!");
    }

    Mat_<float> temp = Mat_<float>(src).clone();

    // 构建 低通、高通 滤波器
    Mat low_filter;
    Mat high_filter;
    waveletFilterD(wname, low_filter, high_filter);

    // 小波变换
    int rows = src.rows;
    int cols = src.cols;
    for (int t = 1; t <= level; t++) {
        // 行小波变换
        for (int r = 0; r < rows; r++) {
            Mat one_row = Mat::zeros(1, cols, temp.type());
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
            Mat one_col = Mat::zeros(rows, 1, temp.type());
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

void IDWT(Mat &src, Mat &dst, const string &wname, int level) {
    if (src.empty()) {
        throw invalid_argument("IDWT(): Input image is empty!");
    }

    Mat_<float> temp = Mat_<float>(src).clone();

    // 构建 低通、高通 滤波器
    Mat lowFilter;
    Mat highFilter;
    waveletFilterR(wname, lowFilter, highFilter);

    // 小波逆变换
    int rows = (int) (src.rows / pow(2.0, level - 1));
    int cols = (int) (src.cols / pow(2.0, level - 1));
    for (int t = 1; t <= level; t++) {
        // 列小波逆变换
        for (int c = 0; c < cols; c++) {
            Mat one_col = Mat::zeros(rows, 1, temp.type());
            for (int i = 0; i < rows; i++) {
                one_col.at<float>(i, 0) = temp.at<float>(i, c);
            }

            one_col = (waveletReconstruct(one_col.t(), lowFilter, highFilter)).t();
            for (int i = 0; i < rows; i++) {
                temp.at<float>(i, c) = one_col.at<float>(i, 0);
            }
        }

        // 行小波逆变换
        for (int r = 0; r < rows; r++) {
            Mat one_row = Mat::zeros(1, cols, temp.type());
            for (int j = 0; j < cols; j++) {
                one_row.at<float>(0, j) = temp.at<float>(r, j);
            }

            one_row = waveletReconstruct(one_row, lowFilter, highFilter);
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

void DCT(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("DCT(): Input image is empty!");
    }

    Mat_<float> temp = Mat_<float>(src).clone();

    // 对输入矩阵进行零填充到合适的尺寸，使用镜像填充
    int M = getOptimalDFTSize(temp.rows);
    int N = getOptimalDFTSize(temp.cols);
    Mat padded;
    copyMakeBorder(temp, padded, 0, M - temp.rows, 0, N - temp.cols, BORDER_REFLECT_101,
                   Scalar::all(0));

    // 离散余弦变换
    dct(padded, temp);

    temp.copyTo(dst);
}

void IDCT(Mat &src, Mat &dst) {
    if (src.empty()) {
        throw invalid_argument("IDCT(): Input image is empty!");
    }

    // 为确保输出图像与原图像的尺寸一致，应初始化并传入正确尺寸的 dst 对象，
    // 如：Mat dst = Mat::zeros(src.size(), src.type());
    if (dst.empty()) {
        throw invalid_argument("IDCT(): Dst size unknown!");
    }

    Mat_<float> temp = Mat_<float>(src).clone();

    // 离散余弦逆变换
    idct(temp, temp);

    // 裁剪至原尺寸
    temp = temp(Rect(0, 0, dst.cols, dst.rows));

    temp.copyTo(dst);
}

void blockDCT(Mat &src, Mat &dst, int block_size) {
    if (src.empty()) {
        throw invalid_argument("blockDCT(): Input image is empty!");
    }

    Mat_<float> temp = Mat_<float>(src).clone();

    // 对输入矩阵进行零填充到合适的尺寸，使用镜像填充
    int M = getOptimalDFTSize(temp.rows);
    int N = getOptimalDFTSize(temp.cols);
    Mat padded;
    copyMakeBorder(temp, padded, 0, M - temp.rows, 0, N - temp.cols, BORDER_REFLECT_101,
                   Scalar::all(0));

    // 分块处理
    for (int i = 0; i < padded.rows; i += block_size) {
        for (int j = 0; j < padded.cols; j += block_size) {
            // 获取块
            Mat block = padded(Rect(j, i, block_size, block_size));

            // 对块进行 dct
            dct(block, block);
        }
    }

    padded.copyTo(dst);
}

void blockIDCT(Mat &src, Mat &dst, int block_size) {
    if (src.empty()) {
        throw invalid_argument("blockIDCT(): Input image is empty!");
    }

    // 为确保输出图像与原图像的尺寸一致，应初始化并传入正确尺寸的 dst 对象，
    // 如：Mat dst = Mat::zeros(src.size(), src.type());
    if (dst.empty()) {
        throw invalid_argument("blockIDCT(): Dst size unknown!");
    }

    Mat_<float> temp = Mat_<float>(src).clone();

    // 分块处理
    for (int i = 0; i < temp.rows; i += block_size)
    {
        for (int j = 0; j < temp.cols; j += block_size)
        {
            // 获取块
            Mat block = temp(Rect(j, i, block_size, block_size));

            // 对块执行 idct
            idct(block, block);
        }
    }

    // 裁剪至原尺寸
    temp = temp(Rect(0, 0, dst.cols, dst.rows));

    temp.copyTo(dst);
}
