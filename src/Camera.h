#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

class Camera
{
private:
    cv::VideoCapture capture;

public:
    Camera();
    ~Camera();

    bool open(int cameraIndex = 0);
    bool isOpened() const;
    bool getFrame(cv::Mat& frame);
    void release();
};