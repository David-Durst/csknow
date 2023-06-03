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
    struct TrajectoryPairData {
        float startDistance, endDistance;
    };

    struct TrajectoryData {
        size_t startTraceIndex, endTraceIndex;
        size_t bestBaselineMatchGeneratedStartTraceIndex, bestBaselineMatchGeneratedEndTraceIndex;
        // for baseline, these are using start/end based on best match generated
        float time, distance;
    };

    class TrajectoryComparison {
    public:
        // per baseline trajectory row per entire generated trajectory data
        vector<vector<TrajectoryPairData>> baselineRowToGeneratedTrajectoryData;

        // per trajectory
        vector<TrajectoryData> baselineData, generatedData;

        void generateTrajectoryStartEnds(vector<TrajectoryData> & trajectoryData,
                                         const csknow::feature_store::TeamFeatureStoreResult & traces);

        TrajectoryComparison(csknow::feature_store::TeamFeatureStoreResult generatedTraces,
                             csknow::feature_store::TeamFeatureStoreResult baselineTraces);
    };
}

#endif //CSKNOW_TRAJECTORY_COMPARISON_H
