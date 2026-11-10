#include "ColorDetector.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/geometry.hpp>

#include <vector>
#include <string>

using namespace cv;
using namespace std;

ColorDetector::ColorDetector()
{
}

void ColorDetector::createTrackbars(const string& windowName)
{
    trackbarWindow = windowName;

    namedWindow(trackbarWindow, WINDOW_NORMAL);

    createTrackbar("H Min", trackbarWindow, &hMin, 179);
    createTrackbar("H Max", trackbarWindow, &hMax, 179);

    createTrackbar("S Min", trackbarWindow, &sMin, 255);
    createTrackbar("S Max", trackbarWindow, &sMax, 255);

    createTrackbar("V Min", trackbarWindow, &vMin, 255);
    createTrackbar("V Max", trackbarWindow, &vMax, 255);
}

void ColorDetector::detect(
    const Mat& frame,
    Mat& mask,
    Mat& result,
    Point& center,
    bool& found
)
{
    Mat hsv;

    found = false;
    center = Point(-1, -1);

    cvtColor(frame, hsv, COLOR_BGR2HSV);

    Scalar lower(hMin, sMin, vMin);
    Scalar upper(hMax, sMax, vMax);

    inRange(hsv, lower, upper, mask);

    result = frame.clone();

    vector<vector<Point>> contours;

    findContours(
        mask,
        contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE
    );

    double maxArea = 500.0;
    int largstContourIndex = -1;

    for (int i = 0; i < static_cast<int>(contours.size()); i++)
    {
        double area = contourArea(contours[i]);

        if (area > maxArea)
        {
            maxArea = area;
            largstContourIndex = i;
        }
    }

    if (largstContourIndex == -1)
        return;
    
    Rect boundingBox = boundingRect(contours[largstContourIndex]);

    center = Point(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);

    found = true;

    rectangle(result, boundingBox, Scalar(0, 255, 0), 2);

    circle(result, center, 5, Scalar(0, 0, 255), -1);

    putText(result, "Green Object", Point(boundingBox.x, boundingBox.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
}