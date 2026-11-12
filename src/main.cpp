#include <iostream>
#include <string>
#include <chrono>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Camera.h"
#include "ColorDetector.h"
#include "MultiObjectTracker.h"

using namespace std;
using namespace cv;

int main()
{
    Camera camera;
    ColorDetector colorDetector;
    MultiObjectTracker multiTracker;

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

    vector<Detection> detections;

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
            chrono::duration<double>(
                currentTime - lastTime
            ).count();

        lastTime = currentTime;

        if (deltaTime > 0.0)
            fps = 1.0 / deltaTime;

        colorDetector.detectAll(
            frame,
            mask,
            colorResult,
            detections
        );

        multiTracker.Update(
            detections,
            deltaTime
        );

        multiTracker.Draw(colorResult);

        string fpsText =
            "FPS: " + to_string(static_cast<int>(fps));

        putText(
            colorResult,
            fpsText,
            Point(20, 40),
            FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar(255, 255, 255),
            2
        );

        string objectCountText =
            "Active tracks: " +
            to_string(multiTracker.GetTrackCount());

        putText(
            colorResult,
            objectCountText,
            Point(20, 75),
            FONT_HERSHEY_SIMPLEX,
            0.7,
            Scalar(0, 255, 255),
            2
        );

        imshow("Original", frame);
        imshow("Multi Object Tracking", colorResult);
        imshow("Color Mask", mask);

        int key = waitKey(1);

        colorDetector.handleKey(key);

        if (key == 27)
            break;
    }

    camera.release();

    destroyAllWindows();

    return 0;
}