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
#define MAX_NAV_CELLS 72000
#define CELL_DIM_WIDTH_DEPTH 16.
#define CELL_DIM_HEIGHT 36.
using std::map;
using std::bitset;
typedef bitset<MAX_NAV_AREAS> AreaBits;
typedef bitset<MAX_NAV_CELLS> CellBits;

struct AreaVisPoint {
    AreaId areaId;
    AABB areaCoordinates;
    Vec3 center;
    AreaBits visibleFromCurPoint = 0;
    AreaBits dangerFromCurPoint = 0;
};

struct CellVisPoint {
    AreaId areaId;
    CellId cellId;
    AABB cellCoordinates;
    Vec3 center;
    CellBits visibleFromCurPoint = 0;
    CellBits dangerFromCurPoint = 0;
};

class VisPoints {
    vector<AreaVisPoint> areaVisPoints;
    vector<CellVisPoint> cellVisPoints;
    map<AreaId, size_t> areaIdToVectorIndex;
    AABB areaBounds;

    void createAreaVisPoints(const nav_mesh::nav_file & navFile);
    void createCellVisPoints();

    void setDangerPoints(const nav_mesh::nav_file & navFile);

public:
    explicit
    VisPoints(const nav_mesh::nav_file & navFile) {
        createAreaVisPoints(navFile);
        createCellVisPoints();
        areaIdToVectorIndex = navFile.m_area_ids_to_indices;
    }

    [[nodiscard]]
    bool isVisibleIndex(size_t src, size_t target) const {
        return areaVisPoints[src].visibleFromCurPoint[target];
    }

    [[nodiscard]]
    bool isVisibleAreaId(AreaId srcId, AreaId targetId) const {
        size_t src = areaIdToVectorIndex.find(srcId)->second, target = areaIdToVectorIndex.find(targetId)->second;
        return areaVisPoints[src].visibleFromCurPoint[target];
    }

    [[nodiscard]]
    AreaBits getVisibilityRelativeToSrc(AreaId srcId) const {
        return areaVisPoints[areaIdToVectorIndex.find(srcId)->second].visibleFromCurPoint;
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
        return areaVisPoints[src].dangerFromCurPoint[target];
    }

    [[nodiscard]] [[maybe_unused]]
    bool isDangerAreaId(AreaId srcId, AreaId targetId) const {
        size_t src = areaIdToVectorIndex.find(srcId)->second, target = areaIdToVectorIndex.find(targetId)->second;
        return areaVisPoints[src].dangerFromCurPoint[target];
    }

    [[nodiscard]]
    AreaBits getDangerRelativeToSrc(AreaId srcId) const {
        return areaVisPoints[areaIdToVectorIndex.find(srcId)->second].dangerFromCurPoint;
    }

    void launchVisPointsCommand(const ServerState & state, bool areas);
    void load(const string & mapsPath, const string & mapName, const nav_mesh::nav_file & navFile);
    [[nodiscard]] const vector<AreaVisPoint> & getVisPoints() const { return areaVisPoints; }
    [[nodiscard]] const vector<CellVisPoint> & getCellVisPoints() const { return cellVisPoints; }
};

#endif //CSKNOW_LOAD_SAVE_VIS_POINTS_H
