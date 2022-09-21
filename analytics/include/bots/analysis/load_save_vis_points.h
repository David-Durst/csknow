//
// Created by steam on 7/8/22.
//

#ifndef CSKNOW_LOAD_SAVE_VIS_POINTS_H
#define CSKNOW_LOAD_SAVE_VIS_POINTS_H
#include "load_data.h"
#include "geometry.h"
#include "geometryNavConversions.h"
#include "navmesh/nav_file.h"
#include "bots/load_save_bot_data.h"
#include <bitset>
#define MAX_NAV_AREAS 2000
using std::map;
using std::bitset;
typedef bitset<MAX_NAV_AREAS> AreaBits;

struct VisPoint {
    AreaId areaId;
    AABB areaCoordinates;
    Vec3 center;
    AreaBits visibleFromCurPoint = 0;
    AreaBits dangerFromCurPoint = 0;
};

class VisPoints {
    vector<VisPoint> visPoints;
    map<AreaId, size_t> areaIdToVectorIndex;

    void setDangerPoints(const nav_mesh::nav_file & navFile);

public:
    explicit
    VisPoints(const nav_mesh::nav_file & navFile) {
        for (const auto & navArea : navFile.m_areas) {
            visPoints.push_back(VisPoint{navArea.get_id(), {vec3tConv(navArea.get_min_corner()), vec3tConv(navArea.get_max_corner())},
                                 vec3tConv(navArea.get_center())});
            visPoints.back().center.z += EYE_HEIGHT;
        }
        std::sort(visPoints.begin(), visPoints.end(),
                  [](const VisPoint & a, const VisPoint & b) { return a.areaId < b.areaId; });
        for (size_t i = 0; i < visPoints.size(); i++) {
            if (navFile.m_areas[i].get_id() != visPoints[i].areaId) {
                std::cout << "vis points loading order wrong" << std::endl;
            }
        }

        areaIdToVectorIndex = navFile.m_area_ids_to_indices;
    }

    [[nodiscard]]
    bool isVisibleIndex(size_t src, size_t target) const {
        return visPoints[src].visibleFromCurPoint[target];
    }

    [[nodiscard]]
    bool isVisibleAreaId(AreaId srcId, AreaId targetId) const {
        size_t src = areaIdToVectorIndex.find(srcId)->second, target = areaIdToVectorIndex.find(targetId)->second;
        return visPoints[src].visibleFromCurPoint[target];
    }

    [[nodiscard]]
    AreaBits getVisibilityRelativeToSrc(AreaId srcId) const {
        return visPoints[areaIdToVectorIndex.find(srcId)->second].visibleFromCurPoint;
    }

    [[maybe_unused]]
    bool isVisiblePlace(AreaId srcId, const string & placeName, const map<string, vector<AreaId>> & placeToArea) {
        AreaBits visibleAreasInPlace;
        if (placeToArea.find(placeName) == placeToArea.end()) {
            return false;
        }
        for (const auto & areaId : placeToArea.find(placeName)->second) {
            visibleAreasInPlace[areaIdToVectorIndex[areaId]] = true;
        }
        visibleAreasInPlace &= getVisibilityRelativeToSrc(srcId);
        return visibleAreasInPlace.any();
    }

    [[nodiscard]]
    bool isDangerIndex(size_t src, size_t target) const {
        return visPoints[src].dangerFromCurPoint[target];
    }

    [[nodiscard]] [[maybe_unused]]
    bool isDangerAreaId(AreaId srcId, AreaId targetId) const {
        size_t src = areaIdToVectorIndex.find(srcId)->second, target = areaIdToVectorIndex.find(targetId)->second;
        return visPoints[src].dangerFromCurPoint[target];
    }

    [[nodiscard]]
    AreaBits getDangerRelativeToSrc(AreaId srcId) const {
        return visPoints[areaIdToVectorIndex.find(srcId)->second].dangerFromCurPoint;
    }

    void launchVisPointsCommand(const ServerState & state);
    void load(string mapsPath, string mapName, const nav_mesh::nav_file & navFile);
    [[nodiscard]] const vector<VisPoint> & getVisPoints() const { return visPoints; }
};

#endif //CSKNOW_LOAD_SAVE_VIS_POINTS_H
