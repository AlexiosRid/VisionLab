#include "ColorDetector.h"
#include <algorithm>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/geometry.hpp>

#include <vector>

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

    setPreset(ColorPreset::GREEN);
}

void ColorDetector::setPreset(ColorPreset preset)
{
    currentPreset = preset;

    switch (preset)
    {
        case ColorPreset::RED:
            hMin = 0;
            hMax = 179;
            sMin = 100;
            sMax = 255;
            vMin = 70;
            vMax = 255;
            break;

        case ColorPreset::GREEN:
            hMin = 35;
            hMax = 85;
            sMin = 50;
            sMax = 255;
            vMin = 50;
            vMax = 255;
            break;

        case ColorPreset::BLUE:
            hMin = 90;
            hMax = 130;
            sMin = 70;
            sMax = 255;
            vMin = 50;
            vMax = 255;
            break;

        case ColorPreset::YELLOW:
            hMin = 20;
            hMax = 35;
            sMin = 80;
            sMax = 255;
            vMin = 100;
            vMax = 255;
            break;

        case ColorPreset::ORANGE:
            hMin = 5;
            hMax = 20;
            sMin = 100;
            sMax = 255;
            vMin = 100;
            vMax = 255;
            break;

        case ColorPreset::PURPLE:
            hMin = 130;
            hMax = 160;
            sMin = 50;
            sMax = 255;
            vMin = 50;
            vMax = 255;
            break;

        case ColorPreset::BLACK:
            hMin = 0;
            hMax = 179;
            sMin = 0;
            sMax = 255;
            vMin = 0;
            vMax = 60;
            break;

        case ColorPreset::WHITE:
            hMin = 0;
            hMax = 179;
            sMin = 0;
            sMax = 50;
            vMin = 180;
            vMax = 255;
            break;

        case ColorPreset::CUSTOM:
            break;
    }

    if (!trackbarWindow.empty())
    {
        setTrackbarPos("H Min", trackbarWindow, hMin);
        setTrackbarPos("H Max", trackbarWindow, hMax);

        setTrackbarPos("S Min", trackbarWindow, sMin);
        setTrackbarPos("S Max", trackbarWindow, sMax);

        setTrackbarPos("V Min", trackbarWindow, vMin);
        setTrackbarPos("V Max", trackbarWindow, vMax);
    }
}

void ColorDetector::handleKey(int key)
{
    switch (key)
    {
        case '1':
            setPreset(ColorPreset::RED);
            break;

        case '2':
            setPreset(ColorPreset::GREEN);
            break;

        case '3':
            setPreset(ColorPreset::BLUE);
            break;

        case '4':
            setPreset(ColorPreset::YELLOW);
            break;

        case '5':
            setPreset(ColorPreset::ORANGE);
            break;

        case '6':
            setPreset(ColorPreset::PURPLE);
            break;

        case '7':
            setPreset(ColorPreset::BLACK);
            break;

        case '8':
            setPreset(ColorPreset::WHITE);
            break;

        case '0':
            setPreset(ColorPreset::CUSTOM);
            break;
    }
}

string ColorDetector::getPresetName() const
{
    switch (currentPreset)
    {
        case ColorPreset::RED:    return "RED";
        case ColorPreset::GREEN:  return "GREEN";
        case ColorPreset::BLUE:   return "BLUE";
        case ColorPreset::YELLOW: return "YELLOW";
        case ColorPreset::ORANGE: return "ORANGE";
        case ColorPreset::PURPLE: return "PURPLE";
        case ColorPreset::BLACK:  return "BLACK";
        case ColorPreset::WHITE:  return "WHITE";
        case ColorPreset::CUSTOM: return "CUSTOM";
    }

    return "UNKNOWN";
}

void ColorDetector::detectAll(const Mat& frame, Mat& mask, Mat& result, vector<Detection>& detections)
{
    detections.clear();

    if (frame.empty())
    {
        mask.release();
        result.release();
        return;
    }

    Mat hsv;

    cvtColor(frame, hsv, COLOR_BGR2HSV);

    if (currentPreset == ColorPreset::RED)
    {
        Mat mask1;
        Mat mask2;

        inRange(hsv, Scalar(0, sMin, vMin), Scalar(10, sMax, vMax), mask1);
        inRange(hsv, Scalar(170, sMin, vMin), Scalar(179, sMax, vMax), mask2);

        bitwise_or(mask1, mask2, mask);
    }
    else
    {
        inRange(hsv, Scalar(hMin, sMin, vMin), Scalar(hMax, sMax, vMax), mask);
    }

    result = frame.clone();

    vector<vector<Point>> contours;

    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    const double minArea = 500.0;

    for (const auto& contour : contours)
    {
        double area = contourArea(contour);

        if (area < minArea)
        {
            continue;
        }
        Rect boundingBox = boundingRect(contour);

        Point center(boundingBox.x / 2 + boundingBox.width / 2,
                     boundingBox.y / 2 + boundingBox.height / 2);

        Detection detection
        {
            center,
            boundingBox,
            area
        };

        detections.push_back(detection);

        rectangle(result, boundingBox, Scalar(0, 255, 0), 2);

        circle(result, center, 5, Scalar(0, 0, 255), 2);
    }

    string label = "Color" + getPresetName() + " | Objects: " + to_string(detections.size());

    int baseline = 0;

    Size textSize = getTextSize(
        label,
        FONT_HERSHEY_SIMPLEX,
        0.6,
        2,
        &baseline
    );

    int margin = 15;

    Point textPosition(
        margin,
        result.rows - margin - baseline
    );

    rectangle(result, Point(margin - 5, result.rows - textSize.height - margin - 5),
        Point(margin + textSize.width + 5, result.rows - margin + baseline),
        Scalar(0, 0, 0),
        FILLED
    );

    putText(
        result,
        label,
        textPosition,
        FONT_HERSHEY_SIMPLEX,
        0.6,
        detections.empty() ? Scalar(0, 0, 255) : Scalar(0, 255, 0),
        2
    );

}