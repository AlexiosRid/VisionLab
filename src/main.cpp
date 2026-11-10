
#include <iostream>
#include <string>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Camera.h"
#include "ImageProcessor.h"
#include "ColorDetector.h"
#include "ObjectTracker.h"

using namespace std;
using namespace cv;

int main()
{
    Camera camera;
    ImageProcessor processor;
    ColorDetector colorDetector;
    ObjectTracker tracker;

    if (!camera.open(0))
    {
        cerr << "Error: camera not found!" << endl;
        return 1;
    }

    cout << "Camera connected successfully!" << endl;

    colorDetector.createTrackbars("HSV Controls");

    Mat frame;
    Mat mask;
    Mat colorResult;

    Point objectCenter;
    bool objectFound = false;

    auto lastTime = chrono::steady_clock::now();
    double fps = 0.0;

    while (true)
    {
        if (!camera.getFrame(frame))
        {
            cerr << "Failed to get frame!" << endl;
            break;
        }

        auto currentTime = chrono::steady_clock::now();

        double deltaTime =
            chrono::duration<double>(currentTime - lastTime).count();

        lastTime = currentTime;

        if (deltaTime > 0)
            fps = 1.0 / deltaTime;

        colorDetector.detect(
            frame,
            mask,
            colorResult,
            objectCenter,
            objectFound
        );

        tracker.Update(
            objectCenter,
            objectFound,
            deltaTime
        );

        tracker.Draw(colorResult);

        string fpsText = "FPS: " +
            to_string(static_cast<int>(fps));

        putText(
            frame,
            fpsText,
            Point(20, 40),
            FONT_HERSHEY_SIMPLEX,
            1.0,
            Scalar(0, 255, 0),
            2
        );

        imshow("Original", frame);
        imshow("Color Detection", colorResult);
        imshow("Color Mask", mask);

        if (waitKey(1) == 27)
            break;
    }

    camera.release();
    destroyAllWindows();

    return 0;
}
