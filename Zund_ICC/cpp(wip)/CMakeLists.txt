cmake_minimum_required(VERSION 3.10)
project(ZundICCPrototype)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(zund_icc_prototype
    main.cpp
    EdgeDetector.cpp
    SearchStrategy.cpp
    EdgeFollower.cpp
    Exporter.cpp
)

target_link_libraries(zund_icc_prototype ${OpenCV_LIBS})
