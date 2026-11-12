
#include "MultiObjectTracker.h"

#include <algorithm>
#include <cmath>


void MultiObjectTracker::Update(
    const vector<Detection>& detections,
    double deltaTime
)
{
    vector<bool> detectionAssigned(
        detections.size(),
        false
    );

    for (Track& track : tracks)
    {
        Point predictedCenter =
            track.GetPredictedCenter(deltaTime);

        int bestDetectionIndex = -1;

        double minDistance = maxAssociationDistance;

        for (size_t i = 0; i < detections.size(); i++)
        {
            if (detectionAssigned[i])
                continue;

            Point detectionCenter = detections[i].center;

            double dx = predictedCenter.x - detectionCenter.x;
            double dy = predictedCenter.y - detectionCenter.y;

            double distance = sqrt(dx * dx + dy * dy);

            if (distance < minDistance)
            {
                minDistance = distance;
                bestDetectionIndex = static_cast<int>(i);
            }
        }

        if (bestDetectionIndex >= 0)
        {
            detectionAssigned[bestDetectionIndex] = true;

            track.Update(
                &detections[bestDetectionIndex],
                deltaTime
            );
        }
        else
        {
            track.Update(nullptr, deltaTime);
        }
    }

    tracks.erase(
        remove_if(
            tracks.begin(),
            tracks.end(),
            [](const Track& track)
            {
                return track.IsExpired();
            }
        ),
        tracks.end()
    );

    for (size_t i = 0; i < detections.size(); i++)
    {
        if (!detectionAssigned[i])
        {
            tracks.emplace_back(
                nextTrackId++,
                detections[i]
            );
        }
    }
}


void MultiObjectTracker::Draw(Mat& frame)
{
    for (Track& track : tracks)
    {
        track.Draw(frame);
    }
}


size_t MultiObjectTracker::GetTrackCount() const
{
    return tracks.size();
}