# VisionLab

**VisionLab** is a C++ computer vision project built with OpenCV.  
The project is designed to explore image processing, color detection, and object tracking in real time.

## Features

- Live camera capture
- FPS counter
- Grayscale image conversion
- Gaussian blur
- Edge detection
- HSV color detection with adjustable trackbars
- Object detection using contours
- Object tracking

## Technologies

- C++20
- OpenCV 5
- CMake
- Ninja
- GCC (MSYS2 UCRT64)

## Project Structure

```text
VisionLab/
├── src/
│   ├── main.cpp
│   ├── Camera.cpp
│   ├── ImageProcessor.cpp
│   ├── ColorDetector.cpp
│   └── ObjectTracker.cpp
├── CMakeLists.txt
├── .gitignore
└── README.md
```

## Requirements

- Windows
- MSYS2 UCRT64
- GCC
- CMake
- Ninja
- OpenCV 5

## Build Instructions

Configure the project:

```powershell
cmake -S . -B build -G Ninja -DOpenCV_DIR="C:/msys64/ucrt64/lib/cmake/opencv5"
```

Build:

```powershell
cmake --build build
```

Run:

```powershell
.\build\VisionLab.exe
```

## Goals

The goal of VisionLab is to build practical computer vision experience in C++ and gradually develop a modular real-time image processing application.

## License

No license has been added yet.
