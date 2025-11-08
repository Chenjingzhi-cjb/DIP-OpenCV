cmake_minimum_required(VERSION 3.19)
project(dip_opencv)

set(CMAKE_CXX_STANDARD 17)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# ------ opencv mingw ------
set(OpenCV_DIR "D:/opencv-470/mingw-build")  # `OpenCVConfig.cmake` 的路径
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_xfeatures2d)

# ------ spdlog mingw lib ------
# cd spdlog && mkdir mingw-build && cd mingw-build
# cmake -G "MinGW Makefiles" -DSPDLOG_BUILD_SHARED=OFF .. && mingw32-make
set(SPDLOG_ROOT "D:/Project/spdlog")
include_directories("${SPDLOG_ROOT}/include")
link_directories("${SPDLOG_ROOT}/mingw-build")
add_definitions(-DSPDLOG_COMPILED_LIB)  # 必须定义此宏

include_directories(include tools)

aux_source_directory(src SRC_LIST)
add_executable(main main.cpp ${SRC_LIST})

target_link_libraries(main ${OpenCV_LIBS} spdlog)  # OpenCV_LIBS 默认为加载 opencv 所有库
