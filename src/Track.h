#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include <deque>
#include <string>
#include <cstddef>

#include "ColorDetector.h"

using namespace std;
using namespace cv;

class Track
{
public:
    Track(int id, const Detection& detection);

    void Update(const Detection* detection, double deltaTime);

    void Draw(Mat& frame);

    bool IsExpired() const;

    int GetId() const;
    int GetMissedFrames() const;

    Point GetCurrentCenter() const;
    Point GetPredictedCenter(double deltaTime) const;
    Rect GetBoundingBox() const;

    double GetSpeed() const;
    string GetDirection() const;

private:
    int id;

    KalmanFilter kalman;
    bool kalmanInitialized = false;

    Point currentCenter = Point(-1, -1);
    Point measurementCenter = Point(-1, -1);

    Point2d velocity = Point2d(0.0, 0.0);

    double speed = 0.0;

    Rect boundingBox;

    deque<Point> trajectory;

    size_t maxTrajectorySize = 40;

    bool hasCurrent = false;
    bool hasMeasurement = false;

    int missedFrames = 0;
    int maxMissedFrames = 15;

    double predictionTime = 0.25;

    string direction = "START";

    void InitializeKalman(const Point& center);

    void UpdateDirection();
};