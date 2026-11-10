#include "ImageProcessor.h"
#include <opencv2/imgproc.hpp>

cv::Mat ImageProcessor::toGrayscale(const cv::Mat& image)
{
    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_BGR2GRAY);
    return result;
}

cv::Mat ImageProcessor::blur(const cv::Mat& image)
{
    cv::Mat result;
    cv::GaussianBlur(image, result, cv::Size(5, 5), 0);
    return result;
}

cv::Mat ImageProcessor::detectEdges(const cv::Mat& image)
{
    cv::Mat gray;
    cv::Mat result;

    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image;

    cv::Canny(gray, result, 100, 200);
    return result;
}