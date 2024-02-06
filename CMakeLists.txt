cmake_minimum_required(VERSION 3.19)
project(dip_opencv)

set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
include_directories(include)
aux_source_directory(src SRC_LIST)

# ------ opencv ------
#set(OpenCV_DIR "C:/xxx/opencv-470/mingw-build")  # 库安装路径，已添加到环境变量则可以不用
find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})  # 一般可以自动找到
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_xfeatures2d)  # 注意添加相应动态库到环境变量

# ------ opencv msvc build (default world) ------
#set(CMAKE_PREFIX_PATH "D:/opencv-470/build")  # 库安装路径
#set(OpenCV_DIR "D:/opencv-470/build/x64/vc16/lib")  # 库静态路径
#find_package(OpenCV REQUIRED)

# ------ opencv msvc msvc-build ------
#set(CMAKE_PREFIX_PATH "D:/opencv-470/msvc-build")  # 库安装路径
#find_package(OpenCV REQUIRED)
#set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)  # 注意添加相应动态库到环境变量

add_executable(main main.cpp ${SRC_LIST})

target_link_libraries(main ${OpenCV_LIBS})  # OpenCV_LIBS 默认为加载 opencv 所有库
