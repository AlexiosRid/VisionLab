#include "Camera.h"

Camera::Camera()
{
}

Camera::~Camera()
{
    release();
}

bool Camera::open(int cameraIndex)
{
    return capture.open(cameraIndex);
}

bool Camera::isOpened() const
{
    return capture.isOpened();
}

bool Camera::getFrame(cv::Mat& frame)
{
    if (!capture.isOpened())
        return false;

    capture >> frame;

    return !frame.empty();
}

void Camera::release()
{
    if (capture.isOpened())
        capture.release();
}