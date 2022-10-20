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
#include "base64.hpp"
#include <bitset>
#include "indices/bitset.h"
constexpr size_t MAX_NAV_AREAS = 2000;
constexpr size_t MAX_NAV_CELLS = 22000;
constexpr size_t NAV_CELLS_PER_ROW = 256;
constexpr double CELL_DIM_WIDTH_DEPTH = 32.;
constexpr double CELL_DIM_HEIGHT = 36.;
using std::map;
using std::bitset;
using std::byte;
typedef csknow::Bitset<MAX_NAV_AREAS> AreaBits;
typedef csknow::Bitset<MAX_NAV_CELLS> CellBits;
typedef array<int64_t, 3> CellDiscreteCoord;

struct AreaVisPoint {
    AreaId areaId;
    AABB areaCoordinates;
    Vec3 center;
    AreaBits visibleFromCurPoint = AreaBits();
    AreaBits dangerFromCurPoint = AreaBits();
};

struct CellVisPoint {
    AreaId areaId;
    CellId cellId;
    CellDiscreteCoord cellDiscreteCoordinates;
    AABB cellCoordinates;
    Vec3 center;
    Vec3 topCenter;
    CellBits visibleFromCurPoint = CellBits();
    CellBits dangerFromCurPoint = CellBits();
};

struct VisCommandRange {
    size_t startRow;
    size_t numRows;
};

class VisPoints {
    vector<AreaVisPoint> areaVisPoints;
    vector<CellVisPoint> cellVisPoints;
    map<AreaId, size_t> areaIdToVectorIndex;
    AABB areaBounds;
    CellDiscreteCoord maxCellNumbersByDim;

    void createAreaVisPoints(const nav_mesh::nav_file & navFile);
    void createCellVisPoints();

    void setDangerPoints(const nav_mesh::nav_file & navFile, bool area);

public:
    explicit
    VisPoints(const nav_mesh::nav_file & navFile) : areaBounds{}, maxCellNumbersByDim{} {
        areaIdToVectorIndex = navFile.m_area_ids_to_indices;
        createAreaVisPoints(navFile);
        createCellVisPoints();
    }

    [[nodiscard]]
    bool isVisibleIndex(size_t src, size_t target, bool area = true) const {
        if (area) {
            return areaVisPoints[src].visibleFromCurPoint[target];
        }
        else {
            return cellVisPoints[src].visibleFromCurPoint[target];
        }
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
            visibleAreasInPlace.set(areaIdToVectorIndex[areaId], true);
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

    static void clearFiles(const ServerState & state);
    bool launchVisPointsCommand(const ServerState & state, bool areas, std::optional<VisCommandRange> range = {});
    bool readVisPointsCommandResult(const ServerState & state, bool areas, std::optional<VisCommandRange> range = {});
    void save(const string & mapsPath, const string & mapName, bool area);
    void load(const string & mapsPath, const string & mapName, bool area, const nav_mesh::nav_file & navFile,
              bool fixSymmetry = false);
    void fix_symmetry(bool area);
    [[nodiscard]] const vector<AreaVisPoint> & getAreaVisPoints() const { return areaVisPoints; }
    [[nodiscard]] const vector<CellVisPoint> & getCellVisPoints() const { return cellVisPoints; }
    [[nodiscard]] string getVisFileName(const string & mapName, bool area, bool compressed) const {
        return mapName + (area ? ".area" : ".cell") + ".vis" + (compressed ? ".gz" : "");
    }
    [[nodiscard]] const AABB & getAreaBounds() const { return areaBounds; };
    [[nodiscard]] const CellDiscreteCoord & getMaxCellNumbersByDim() const { return maxCellNumbersByDim; };
};

#endif //CSKNOW_LOAD_SAVE_VIS_POINTS_H
