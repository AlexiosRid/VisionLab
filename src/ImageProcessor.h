#pragma once

#include <opencv2/core.hpp>

class ImageProcessor
{
public:
    cv::Mat toGrayscale(const cv::Mat& image);
    cv::Mat blur(const cv::Mat& image);
    cv::Mat detectEdges(const cv::Mat& image);
};