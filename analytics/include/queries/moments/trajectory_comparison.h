//
// Created by durst on 6/2/23.
//

#ifndef CSKNOW_TRAJECTORY_COMPARISON_H
#define CSKNOW_TRAJECTORY_COMPARISON_H

#include <set>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <list>
#include <map>
#include "bots/analysis/feature_store_team.h"

namespace csknow::trajectory_comparison {
    struct BaselineTraceRowToGeneratedTrajectoryData {
        bool valid;
        float startDistance, endDistance;
        std::map<int, int> ctGeneratedColumnIndexToBaselineColumnIndex, tGeneratedColumnIndexToBaselineColumnIndex;
    };

    struct TrajectoryData {
        size_t startTraceIndex, endTraceIndex;
    };

    struct BaselineTrajectoryToGeneratedTrajectoryData {
        size_t bestMatchBaselineStartTraceIndex, bestMatchBaselineEndTraceIndex;
        float baselineTime, baselineDistance;
        float generatedTime, generatedDistance;
    };

    class TrajectoryComparison {
    public:
        // per baseline trajectory row in trace per entire generated trajectory data
        vector<vector<BaselineTraceRowToGeneratedTrajectoryData>> baselineTraceRowToGeneratedTrajectoryData;

        // per trajectory
        vector<TrajectoryData> baselineData, generatedData;

        // per baseline generated trajectory pair
        vector<vector<BaselineTrajectoryToGeneratedTrajectoryData>> baselineTrajectoryToGeneratedTrajectoryData;

        void generateTrajectoryStartEnds(vector<TrajectoryData> & trajectoryData,
                                         const csknow::feature_store::TeamFeatureStoreResult & traces);

        TrajectoryComparison(csknow::feature_store::TeamFeatureStoreResult generatedTraces,
                             csknow::feature_store::TeamFeatureStoreResult baselineTraces);
    };
}

#endif //CSKNOW_TRAJECTORY_COMPARISON_H
