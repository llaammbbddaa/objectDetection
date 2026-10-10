cmake_minimum_required(VERSION 3.10)
project(yolo26_detect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.11 REQUIRED COMPONENTS core imgproc imgcodecs dnn)

add_executable(yolo26_detect main.cpp)
target_include_directories(yolo26_detect PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolo26_detect PRIVATE ${OpenCV_LIBS})
