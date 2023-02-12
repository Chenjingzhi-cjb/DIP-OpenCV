cmake_minimum_required(VERSION 3.19)
project(dip_opencv)

set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
include_directories(include)
aux_source_directory(src SRC_LIST)

# opencv
find_package(OpenCV REQUIRED)
#set(OpenCV_DIR "C:/Users/12865/Desktop/opencv-455/mingw-build")  # 注意修改
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
#include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp ${SRC_LIST})

# opencv
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
target_link_libraries(main ${OpenCV_LIBS})
