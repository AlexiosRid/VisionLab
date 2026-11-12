#pragma once

#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

enum class ColorPreset
{
    RED,
    GREEN,
    BLACK,
    WHITE,
    BLUE,
    YELLOW,
    ORANGE,
    CUSTOM,
    PURPLE
};

class ColorDetector
{
public:
    ColorDetector();

    void createTrackbars(const std::string& windowName);

    void setPreset(ColorPreset preset);

    void handleKey(int key);

    void detect(
        const Mat& frame,
        Mat& mask,
        Mat& result,
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

    string trackbarWindow;

    ColorPreset currentPreset = ColorPreset::GREEN;

    string getPresetName() const;
};