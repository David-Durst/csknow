//
// Created by steam on 7/9/22.
//

#include "bots/analysis/load_save_vis_points.h"
#include <filesystem>

void VisPoints::createAreaVisPoints(const nav_mesh::nav_file & navFile) {
    for (const auto & navArea : navFile.m_areas) {
        areaVisPoints.push_back(AreaVisPoint{navArea.get_id(), {vec3tConv(navArea.get_min_corner()), vec3tConv(navArea.get_max_corner())},
                                             vec3tConv(navArea.get_center())});
        areaVisPoints.back().center.z += EYE_HEIGHT;
    }
    std::sort(areaVisPoints.begin(), areaVisPoints.end(),
              [](const AreaVisPoint & a, const AreaVisPoint & b) { return a.areaId < b.areaId; });
    for (size_t i = 0; i < areaVisPoints.size(); i++) {
        if (navFile.m_areas[i].get_id() != areaVisPoints[i].areaId) {
            std::cout << "vis points loading order wrong" << std::endl;
        }
    }
}

void VisPoints::createCellVisPoints() {
    areaBounds = {{
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max()
    }, {
        -1. * std::numeric_limits<double>::max(),
        -1. * std::numeric_limits<double>::max(),
        -1. * std::numeric_limits<double>::max()
    }};
    for (const auto & areaVisPoint : areaVisPoints) {
        areaBounds.min = min(areaBounds.min, areaVisPoint.areaCoordinates.min);
        areaBounds.max = max(areaBounds.max, areaVisPoint.areaCoordinates.max);
    }
    // increment max z by player height so get cells up to player height
    areaBounds.max.z += PLAYER_HEIGHT;

    // for every nav area, find the nav cells aligned based on area bounds
    for (const auto & areaVisPoint : areaVisPoints) {
        // extend areaVisPoint height by player height so get vis points up to player height
        AABB extendedAABB = areaVisPoint.areaCoordinates;
        extendedAABB.max.z += PLAYER_HEIGHT;
        // get smallest and largest cell mins that overlap with area
        IVec3 minCellMinInArea = vec3ToIVec3((extendedAABB.min - areaBounds.min));
        minCellMinInArea.x /= CELL_DIM_WIDTH_DEPTH;
        minCellMinInArea.y /= CELL_DIM_WIDTH_DEPTH;
        minCellMinInArea.z /= CELL_DIM_HEIGHT;
        IVec3 maxCellMinInArea = vec3ToIVec3((extendedAABB.max - areaBounds.min));
        maxCellMinInArea.x /= CELL_DIM_WIDTH_DEPTH;
        maxCellMinInArea.y /= CELL_DIM_WIDTH_DEPTH;
        maxCellMinInArea.z /= CELL_DIM_HEIGHT;

        // then core all cells in between, if center is in area, then assign it to this area
        for (int64_t curXId = minCellMinInArea.x; curXId <= maxCellMinInArea.x; curXId++) {
            for (int64_t curYId = minCellMinInArea.y; curYId <= maxCellMinInArea.y; curYId++) {
                for (int64_t curZId = minCellMinInArea.z; curZId <= maxCellMinInArea.z; curZId++) {
                    Vec3 cellMin = areaBounds.min +
                        Vec3{CELL_DIM_WIDTH_DEPTH * curXId, CELL_DIM_WIDTH_DEPTH * curYId, CELL_DIM_HEIGHT * curZId};
                    Vec3 cellMax = cellMin + Vec3{CELL_DIM_WIDTH_DEPTH, CELL_DIM_WIDTH_DEPTH, CELL_DIM_HEIGHT};
                    Vec3 cellCenter = (cellMin + cellMax) / 2.;
                    /*
                    if (pointInRegionMaxInclusive(AABB{cellMin, cellMax}, {378.619, 226., cellCenter.z})) {
                        int x = 1;
                        (void) x;
                    }
                     */
                    if (pointInRegionMaxInclusive(extendedAABB, cellCenter)) {
                        cellVisPoints.push_back({
                            areaVisPoint.areaId,
                            static_cast<CellId>(cellVisPoints.size()),
                            {cellMin, cellMax},
                            cellCenter,
                        });
                    }
                }
            }
        }

    }

}

void VisPoints::launchVisPointsCommand(const ServerState & state, bool areas) {
    string visPointsFileName = "vis_points.csv";
    string visPointsFilePath = state.dataPath + "/" + visPointsFileName;
    string tmpVisPointsFileName = "vis_points.csv.tmp.write";
    string tmpVisPointsFilePath = state.dataPath + "/" + tmpVisPointsFileName;

    std::stringstream visPointsStream;

    if (areas) {
        for (const auto & visPoint : areaVisPoints) {
            visPointsStream << visPoint.center.toCSV() << std::endl;
        }
    }
    else {
        for (const auto & visPoint : cellVisPoints) {
            visPointsStream << visPoint.center.toCSV() << std::endl;
        }
    }

    std::ofstream fsVisPoints(tmpVisPointsFilePath);
    fsVisPoints << visPointsStream.str();
    fsVisPoints.close();

    std::filesystem::rename(tmpVisPointsFilePath, visPointsFilePath);

    state.saveScript({"sm_queryAllVisPointPairs"});
}

void VisPoints::load(const string & mapsPath, const string & mapName, const nav_mesh::nav_file & navFile) {
    string visValidFileName = mapName + ".vis";
    string visValidFilePath = mapsPath + "/" + visValidFileName;

    std::ifstream fsVisValid(visValidFilePath);

    if (std::filesystem::exists(visValidFilePath)) {
        string visValidBuf;
        size_t i = 0;
        while (getline(fsVisValid, visValidBuf)) {
            if (visValidBuf.size() != areaVisPoints.size()) {
                throw std::runtime_error("wrong number of cols in vis valid file's line " + std::to_string(i));
            }

            areaVisPoints[i].visibleFromCurPoint.reset();
            for (size_t j = 0; j < visValidBuf.size(); j++) {
                if (visValidBuf[j] != 't' && visValidBuf[j] != 'f') {
                    throw std::runtime_error("invalid char " + std::to_string(visValidBuf[j]) +
                        " vis valid file's line " + std::to_string(i) + " col " + std::to_string(j));
                }
                areaVisPoints[i].visibleFromCurPoint[j] = visValidBuf[j] == 't';
            }
            i++;
        }
        if (i != areaVisPoints.size()) {
            throw std::runtime_error("wrong number of rows in vis valid file");
        }
    }
    else {
        throw std::runtime_error("no vis valid file");
    }

    // after loading, max sure to or all together, want matrix to be full so can export any row
    for (size_t i = 0; i < areaVisPoints.size(); i++) {
        // set diagonal to true, can see yourself
        areaVisPoints[i].visibleFromCurPoint[i] = true;
        for (size_t j = 0; j < areaVisPoints.size(); j++) {
            areaVisPoints[i].visibleFromCurPoint[j] =
                areaVisPoints[i].visibleFromCurPoint[j] | areaVisPoints[j].visibleFromCurPoint[i];
            areaVisPoints[j].visibleFromCurPoint[i] = areaVisPoints[i].visibleFromCurPoint[j];
        }
    }

    // after completing visibility, compute danger
    setDangerPoints(navFile);
}

void VisPoints::setDangerPoints(const nav_mesh::nav_file & navFile) {
    for (size_t srcArea = 0; srcArea < areaVisPoints.size(); srcArea++) {
        for (size_t dangerArea = 0; dangerArea < areaVisPoints.size(); dangerArea++) {
            if (isVisibleIndex(srcArea, dangerArea)) {
                for (size_t i = 0; i < navFile.connections_area_length[dangerArea]; i++) {
                    size_t conAreaIndex = navFile.connections[navFile.connections_area_start[dangerArea] + i];
                    if (!isVisibleIndex(srcArea, conAreaIndex)) {
                        areaVisPoints[srcArea].dangerFromCurPoint[dangerArea] = true;
                        break;
                    }
                }
            }
        }
    }
}

