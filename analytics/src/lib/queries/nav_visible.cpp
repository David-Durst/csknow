//
// Created by durst on 1/11/22.
//

#include "queries/nav_visible.h"

NavVisibleResult queryNavVisible(const VisPoints & visPoints, const string & overlayLabelsQuery) {
    NavVisibleResult result(overlayLabelsQuery);
    result.numAreas = static_cast<int64_t>(visPoints.getVisPoints().size());
    result.coordinate = {};
    for (const auto & visPoint : visPoints.getVisPoints()) {
        result.coordinate.push_back(visPoint.areaCoordinates);
    }
    result.visibleMatrix.resize(result.numAreas * result.numAreas, false);

    for (int64_t i = 0; i < result.numAreas; i++) {
        for (int64_t j = 0; j < result.numAreas; j++) {
            result.visibleMatrix[i * result.numAreas + j] = visPoints.isVisibleIndex(i, j);
        }
    }

    result.size = result.numAreas;
    return result;
}
