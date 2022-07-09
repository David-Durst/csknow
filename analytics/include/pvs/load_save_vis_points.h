//
// Created by steam on 7/8/22.
//

#ifndef CSKNOW_LOAD_SAVE_VIS_POINTS_H
#define CSKNOW_LOAD_SAVE_VIS_POINTS_H
#include "load_data.h"
#include "geometry.h"
#include "geometryNavConversions.h"
#include "navmesh/nav_file.h"
using std::map;
typedef uint32_t AreaId;

struct VisPoint {
    AreaId areaId;
    Vec3 center;
    vector<bool> visibleFromCurPoint;
};

class VisPoints {
    vector<VisPoint> visPoints;

public:
    VisPoints(nav_mesh::nav_file navFile) {
        for (const auto & navArea : navFile.m_areas) {
            visPoints.push_back({navArea.get_id(), vec3tConv(navArea.get_center()), {}});
        }
    }

    bool isVisible(AreaId src, AreaId target) const {
        return visPoints[std::min(src, target)].visibleFromCurPoint[std::max(src, target)];
    }

    void launchVisPointsCommand();
    void loadVisPoints(bool loadFromCSGOServer = false);
};

#endif //CSKNOW_LOAD_SAVE_VIS_POINTS_H
