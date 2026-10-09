cmake_minimum_required(VERSION 3.10)
project(yolo11_detect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.5.4 REQUIRED COMPONENTS core imgproc imgcodecs dnn)

add_executable(yolo11_detect main.cpp)
target_include_directories(yolo11_detect PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo11_detect PRIVATE ${OpenCV_LIBS})
