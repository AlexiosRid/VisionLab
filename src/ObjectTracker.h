#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <deque>
#include <string>

using namespace std;
using namespace cv;

class ObjectTracker
{
private:
    KalmanFilter kalman;
    bool kalmanInitialized = false;

    Point currentCenter = Point(-1, -1);
    Point measurementCenter = Point(-1, -1);

    Point2d velocity = Point2d(0.0, 0.0);
    double speed = 0.0;

    deque<Point> trajectory;

    size_t maxTrajectorySize = 40;

    bool hasCurrent = false;
    bool hasMeasurement = false;

    int missedFrames = 0;
    int maxMissedFrames = 15;

    double predictionTime = 0.25;

    string direction = "NO OBJECT";

    void InitializeKalman(const Point& center);
    void ResetTracking();

    void UpdateDirection();

public:
    void Update(const Point& center, bool found, double deltaTime);

    void Draw(Mat& frame);
};