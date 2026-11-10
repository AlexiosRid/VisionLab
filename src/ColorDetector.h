#pragma once

#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;

class ColorDetector
{
public:
    ColorDetector();

    void createTrackbars(const std::string& windowName);

    void detect(
        const cv::Mat& frame,
        cv::Mat& mask,
        cv::Mat& result,
        Point& center,
        bool& found
    );

private:
    int hMin = 35;
    int hMax = 85;

    int sMin = 50;
    int sMax = 255;

    int vMin = 50;
    int vMax = 255;

    std::string trackbarWindow;
};