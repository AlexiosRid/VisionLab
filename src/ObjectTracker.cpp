
#include "ObjectTracker.h"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;


void ObjectTracker::InitializeKalman(const Point& center)
{
    kalman.init(4, 2, 0, CV_32F);

    kalman.transitionMatrix = Mat::eye(4, 4, CV_32F);

    kalman.measurementMatrix = Mat::zeros(2, 4, CV_32F);

    kalman.measurementMatrix.at<float>(0, 0) = 1.0f;
    kalman.measurementMatrix.at<float>(1, 1) = 1.0f;

    kalman.measurementNoiseCov =
        Mat::eye(2, 2, CV_32F) * 3.0f;

    kalman.processNoiseCov = Mat::zeros(4, 4, CV_32F);

    kalman.errorCovPost = Mat::eye(4, 4, CV_32F) * 10.0f;


    kalman.statePost.at<float>(0) = static_cast<float>(center.x);
    kalman.statePost.at<float>(1) = static_cast<float>(center.y);
    kalman.statePost.at<float>(2) = 0.0f;
    kalman.statePost.at<float>(3) = 0.0f;

    kalmanInitialized = true;

    currentCenter = center;
    measurementCenter = center;

    velocity = Point2d(0.0, 0.0);
    speed = 0.0;

    hasCurrent = true;
    hasMeasurement = true;

    missedFrames = 0;

    direction = "START";

    trajectory.clear();
    trajectory.push_back(center);
}


void ObjectTracker::ResetTracking()
{
    kalmanInitialized = false;

    hasCurrent = false;
    hasMeasurement = false;

    missedFrames = 0;

    currentCenter = Point(-1, -1);

    velocity = Point2d(0.0, 0.0);
    speed = 0.0;

    direction = "NO OBJECT";

    trajectory.clear();
}


void ObjectTracker::UpdateDirection()
{
    if (speed < 5.0)
    {
        direction = "STOP";
        return;
    }

    if (abs(velocity.x) > abs(velocity.y))
    {
        direction = velocity.x > 0 ? "RIGHT" : "LEFT";
    }
    else
    {
        direction = velocity.y > 0 ? "DOWN" : "UP";
    }
}



void ObjectTracker::Update(
    const Point& center,
    bool found,
    double deltaTime
)
{
    hasMeasurement = found;

    if (!found && !kalmanInitialized)
    {
        direction = "NO OBJECT";
        hasCurrent = false;
        return;
    }

    if (!kalmanInitialized)
    {
        InitializeKalman(center);
        return;
    }

    if (deltaTime <= 0.0)
        deltaTime = 1.0 / 30.0;

    deltaTime = min(deltaTime, 0.2);

    kalman.transitionMatrix.at<float>(0, 2) =
        static_cast<float>(deltaTime);

    kalman.transitionMatrix.at<float>(1, 3) =
        static_cast<float>(deltaTime);

    float dt = static_cast<float>(deltaTime);

    float q = 175.0f;

    float dt2 = dt * dt;
    float dt3 = dt2 * dt;
    float dt4 = dt3 * dt;

    kalman.processNoiseCov = Mat::zeros(4, 4, CV_32F);

    kalman.processNoiseCov.at<float>(0, 0) = q * dt4 / 4.0f;
    kalman.processNoiseCov.at<float>(0, 2) = q * dt3 / 2.0f;
    kalman.processNoiseCov.at<float>(2, 0) = q * dt3 / 2.0f;
    kalman.processNoiseCov.at<float>(2, 2) = q * dt2;

    kalman.processNoiseCov.at<float>(1, 1) = q * dt4 / 4.0f;
    kalman.processNoiseCov.at<float>(1, 3) = q * dt3 / 2.0f;
    kalman.processNoiseCov.at<float>(3, 1) = q * dt3 / 2.0f;
    kalman.processNoiseCov.at<float>(3, 3) = q * dt2;

    Mat predicted = kalman.predict();

    Mat estimated;

    if (found)
    {
        Mat measurement = Mat::zeros(2, 1, CV_32F);

        measurement.at<float>(0) =
            static_cast<float>(center.x);

        measurement.at<float>(1) =
            static_cast<float>(center.y);

        estimated = kalman.correct(measurement);

        measurementCenter = center;

        missedFrames = 0;
    }
    else
    {
        estimated = predicted;

        missedFrames++;
    }

    float x = estimated.at<float>(0);
    float y = estimated.at<float>(1);

    float vx = estimated.at<float>(2);
    float vy = estimated.at<float>(3);

    currentCenter = Point(
        cvRound(x),
        cvRound(y)
    );

    velocity = Point2d(vx, vy);

    speed = sqrt(
        velocity.x * velocity.x +
        velocity.y * velocity.y
    );

    hasCurrent = true;

    UpdateDirection();

    trajectory.push_back(currentCenter);

    if (trajectory.size() > maxTrajectorySize)
        trajectory.pop_front();

    if (missedFrames > maxMissedFrames)
    {
        ResetTracking();
    }
}

void ObjectTracker::Draw(Mat& frame)
{
    if (trajectory.size() >= 2)
    {
        auto previous = trajectory.begin();
        auto current = next(previous);

        for (; current != trajectory.end(); ++previous, ++current)
        {
            line(
                frame,
                *previous,
                *current,
                Scalar(255, 0, 0),
                2
            );
        }
    }

    if (hasCurrent)
    {
        circle(
            frame,
            currentCenter,
            6,
            Scalar(255, 255, 0),
            FILLED
        );

        if (hasMeasurement)
        {
            circle(
                frame,
                measurementCenter,
                4,
                Scalar(0, 255, 0),
                FILLED
            );
        }

        if (speed > 5.0)
        {
            Point predictedPosition(
                cvRound(currentCenter.x + velocity.x * predictionTime),
                cvRound(currentCenter.y + velocity.y * predictionTime)
            );

            arrowedLine(
                frame,
                currentCenter,
                predictedPosition,
                Scalar(0, 255, 255),
                2
            );

            circle(
                frame,
                predictedPosition,
                7,
                Scalar(0, 255, 255),
                2
            );
        }

        string positionText =
            "X: " + to_string(currentCenter.x) +
            " Y: " + to_string(currentCenter.y);

        putText(
            frame,
            positionText,
            Point(20, 140),
            FONT_HERSHEY_SIMPLEX,
            0.7,
            Scalar(0, 255, 0),
            2
        );
    }

    putText(
        frame,
        "Direction: " + direction,
        Point(20, 30),
        FONT_HERSHEY_SIMPLEX,
        0.7,
        Scalar(0, 255, 255),
        2
    );

    putText(
        frame,
        "Speed: " + to_string(static_cast<int>(speed)) + " px/s",
        Point(20, 60),
        FONT_HERSHEY_SIMPLEX,
        0.7,
        Scalar(0, 255, 255),
        2
    );

    string trackingStatus =
        hasMeasurement ? "DETECTED" :
        hasCurrent ? "PREDICTING" : "LOST";

    putText(
        frame,
        trackingStatus,
        Point(20, 100),
        FONT_HERSHEY_SIMPLEX,
        0.7,
        Scalar(255, 255, 255),
        2
    );
}