//
// Created by steam on 7/26/22.
//

#include "queries/nav_danger.h"

NavDangerResult queryNavDanger(const VisPoints & visPoints, const nav_mesh::nav_file & navFile) {
    NavDangerResult result;
    result.numAreas = visPoints.getVisPoints().size();
    result.coordinate = {};
    for (const auto & visPoint : visPoints.getVisPoints()) {
        result.coordinate.push_back(visPoint.areaCoordinates);
    }
    result.dangerMatrix.resize(result.numAreas * result.numAreas, false);

    for (int64_t i = 0; i < result.numAreas; i++) {
        for (int64_t j = 0; j < result.numAreas; j++) {
            result.dangerMatrix[i * result.numAreas + j] = visPoints.isDangerIndex(i, j);
        }
    }

    result.size = result.numAreas;
    return result;
}
