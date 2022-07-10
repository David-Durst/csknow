//
// Created by durst on 1/11/22.
//

#include "queries/nav_visible.h"

NavVisibleResult queryNavVisible(const VisPoints & visPoints) {
    NavVisibleResult result;
    int64_t i = 0;
    result.numAreas = visPoints.getVisPoints().size();
    result.coordinate = {};
    for (const auto & visPoint : visPoints.getVisPoints()) {
        result.coordinate.push_back(visPoint.areaCoordinates);
    }
    result.visibleMatrix.resize(result.numAreas * result.numAreas, std::numeric_limits<double>::max());

    // then compute nearest adjacency matrix with Warshall's Algorithm
    for (int64_t i = 0; i < result.numAreas; i++) {
        for (int64_t j = 0; j < result.numAreas; j++) {
            result.visibleMatrix[i * result.numAreas + j] = visPoints.isVisibleIndex(i, j);
        }
    }

    result.size = result.numAreas;
    return result;
}
