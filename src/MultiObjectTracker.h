#pragma once

#include <vector>

#include "Track.h"

class MultiObjectTracker
{
public:
    void Update(
        const vector<Detection>& detections,
        double deltaTime
    );

    void Draw(Mat& frame);

    size_t GetTrackCount() const;

private:
    vector<Track> tracks;

    int nextTrackId = 1;

    double maxAssociationDistance = 80.0;
};