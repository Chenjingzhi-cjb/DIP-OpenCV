#include <windows.h>

#include <iostream>
#include "head.h"


int main() {
//    Mat image = imread(R"(../image/building-600by600.tif)");
//
//    namedWindow("img", WINDOW_AUTOSIZE);
//    imshow("img", image);
//    waitKey(0);

    HANDLE file_handle;  // 文件句柄
    WIN32_FIND_DATA file_info;  // 文件信息

    std::string folder_path = R"(\)";
    if ((file_handle = FindFirstFile((folder_path + "*").c_str(), &file_info)) !=
        INVALID_HANDLE_VALUE) {
        do {
            if ((file_info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
                std::string image_path = folder_path + file_info.cFileName;
                std::cout << "image path: " << image_path << std::endl;

                Mat image = imread(image_path, 0);
                cv::resize(image, image, Size(image.cols / 4, image.rows / 4));

                double S = calcSharpnessOldOpt(&image, 4);
                std::cout << "value: " << S << std::endl;
            }
        } while (FindNextFile(file_handle, &file_info) != 0);  // 处理下一个，存在则返回值不为 0
        FindClose(file_handle);
    }

    return 0;
}

