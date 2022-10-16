//
// Created by durst on 1/11/22.
//

#include "queries/nav_visible.h"

NavVisibleResult queryNavVisible(const VisPoints & visPoints, const string & overlayLabelsQuery, bool area) {
    NavVisibleResult result(overlayLabelsQuery, area, visPoints);
    result.coordinate = {};
    if (area) {
        result.numPoints = static_cast<int64_t>(visPoints.getAreaVisPoints().size());
        for (const auto & visPoint : visPoints.getAreaVisPoints()) {
            result.coordinate.push_back(visPoint.areaCoordinates);
        }
    }
    else {
        result.numPoints = static_cast<int64_t>(visPoints.getCellVisPoints().size());
        for (const auto & visPoint : visPoints.getCellVisPoints()) {
            result.coordinate.push_back(visPoint.cellCoordinates);
        }
    }
    result.visibleMatrix.resize(result.numPoints * result.numPoints, false);
    result.size = result.numPoints;
    return result;
}
