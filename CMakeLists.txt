cmake_minimum_required(VERSION 3.20)

project(VisionLab LANGUAGES CXX)

set(OpenCV_DIR "C:/msys64/ucrt64/lib/cmake/opencv5")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)

add_executable(VisionLab
    src/main.cpp
    src/Camera.cpp
    src/ImageProcessor.cpp
    src/ColorDetector.cpp
    src/ObjectTracker.cpp
    src/Track.cpp
    src/MultiObjectTracker.cpp
)

target_include_directories(VisionLab PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(VisionLab PRIVATE
    ${OpenCV_LIBS}
)
