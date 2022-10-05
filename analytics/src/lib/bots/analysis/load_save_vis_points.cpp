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
                        Vec3{CELL_DIM_WIDTH_DEPTH * static_cast<double>(curXId),
                             CELL_DIM_WIDTH_DEPTH * static_cast<double>(curYId),
                             CELL_DIM_HEIGHT * static_cast<double>(curZId)};
                    Vec3 cellMax = cellMin + Vec3{CELL_DIM_WIDTH_DEPTH, CELL_DIM_WIDTH_DEPTH, CELL_DIM_HEIGHT};
                    Vec3 cellCenter = (cellMin + cellMax) / 2.;
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

bool VisPoints::launchVisPointsCommand(const ServerState & state, bool areas, std::optional<VisCommandRange> range) {
    string visPointsFileName = "vis_points.csv";
    string visPointsFilePath = state.dataPath + "/" + visPointsFileName;
    string tmpVisPointsFileName = "vis_points.csv.tmp.write";
    string tmpVisPointsFilePath = state.dataPath + "/" + tmpVisPointsFileName;

    if (!range || range->startRow == 0) {
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
    }


    if (!range) {
        range = {0, areas ? areaVisPoints.size() : cellVisPoints.size()};
    }

    string visRangeFileName = "vis_range.csv";
    string visRangeFilePath = state.dataPath + "/" + visRangeFileName;
    string tmpVisRangeFileName = "vis_range.csv.tmp.write";
    string tmpVisRangeFilePath = state.dataPath + "/" + tmpVisRangeFileName;

    std::ofstream fsVisRange(tmpVisRangeFilePath);
    fsVisRange << range->startRow << "," << range->numRows;
    fsVisRange.close();

    std::filesystem::rename(tmpVisRangeFilePath, visRangeFilePath);
    return state.saveScript({"sm_queryRangeVisPointPairs"});
}

bool VisPoints::readVisPointsCommandResult(const ServerState &state, bool areas, std::optional<VisCommandRange> range) {
    string visValidFileName = "vis_valid.csv";
    string visValidFilePath = state.dataPath + "/" + visValidFileName;
    string tmpVisValidFileName = "vis_valid.csv.tmp.write";
    string tmpVisValidFilePath = state.dataPath + "/" + tmpVisValidFileName;

    std::ifstream fsVisValid(visValidFilePath);

    if (std::filesystem::exists(visValidFilePath)) {
        string visValidBuf;
        while (getline(fsVisValid, visValidBuf)) {
            std::stringstream visValidLineStream(visValidBuf);
            string visPointBuf[2];
            getline(visValidLineStream, visPointBuf[0], ',');
            getline(visValidLineStream, visPointBuf[1], ',');

            int64_t visPoints[2] = {stol(visPointBuf[0]), stol(visPointBuf[1])};

            if (visPoints[0] < static_cast<int64_t>(range->startRow) ||
                visPoints[0] >= static_cast<int64_t>(range->startRow + range->numRows)) {
                throw std::runtime_error("point " + std::to_string(visPoints[0]) + " not in range [" +
                    std::to_string(range->startRow) + ", " + std::to_string(range->startRow + range->numRows) +
                    ")");
            }

            auto maxRow = static_cast<int64_t>(areas ? areaVisPoints.size() : cellVisPoints.size());
            if (visPoints[1] < 0 || visPoints[1] >= maxRow) {
                throw std::runtime_error("point " + std::to_string(visPoints[1]) + " not valid point index of " +
                                         std::to_string(maxRow));
            }

            if (areas) {
                areaVisPoints[visPoints[0]].visibleFromCurPoint[visPoints[1]] = true;
            }
            else {
                cellVisPoints[visPoints[0]].visibleFromCurPoint[visPoints[1]] = true;
            }
        }
        return true;
    }
    else {
        return false;
    }
}

void VisPoints::save(const string & mapsPath, const string & mapName, bool area) {
    string visValidFileName = mapName + (area ? "_area" : "_cell") + ".vis";
    string visValidFilePath = mapsPath + "/" + visValidFileName;

    std::ofstream fsVisValid(visValidFilePath, std::ios::out | std::ios::binary);
    if (area) {
        for (const auto & areaVisPoint : areaVisPoints) {
            vector<uint32_t> visibleSparseIds = bitsetToSparseIds(areaVisPoint.visibleFromCurPoint);
            fsVisValid.write(reinterpret_cast<const char*>(visibleSparseIds.size()),
                             static_cast<std::streamsize>(sizeof(size_t)));
            fsVisValid.write(reinterpret_cast<const char*>(visibleSparseIds.data()),
                             static_cast<std::streamsize>(visibleSparseIds.size() * sizeof(uint32_t)));
        }
    }
    else {
        for (const auto & cellVisPoint : cellVisPoints) {
            vector<uint32_t> visibleSparseIds = bitsetToSparseIds(cellVisPoint.visibleFromCurPoint);
            size_t size = visibleSparseIds.size();
            fsVisValid.write(reinterpret_cast<const char*>(&size),
                             static_cast<std::streamsize>(sizeof(size_t)));
            fsVisValid.write(reinterpret_cast<const char*>(visibleSparseIds.data()),
                             static_cast<std::streamsize>(visibleSparseIds.size() * sizeof(uint32_t)));
        }
    }
    fsVisValid.close();
}

void VisPoints::load(const string & mapsPath, const string & mapName, bool area, const nav_mesh::nav_file & navFile) {
    string visValidFileName = mapName + (area ? "_area" : "_cell") + ".vis";
    string visValidFilePath = mapsPath + "/" + visValidFileName;

    std::ifstream fsVisValid(visValidFilePath, std::ios::in | std::ios::binary);
    if (area) {
        for (auto & areaVisPoint : areaVisPoints) {
            size_t size;
            fsVisValid.read(reinterpret_cast<char*>(&size),
                            static_cast<std::streamsize>(sizeof(size_t)));
            vector<uint32_t> visibleSparseIds(size);
            fsVisValid.read(reinterpret_cast<char*>(visibleSparseIds.data()),
                            static_cast<std::streamsize>(size * sizeof(uint32_t)));
            sparseIdsToBitset(visibleSparseIds, areaVisPoint.visibleFromCurPoint);
        }
    }
    else {
        for (auto & cellVisPoint : cellVisPoints) {
            size_t size;
            fsVisValid.read(reinterpret_cast<char*>(&size),
                            static_cast<std::streamsize>(sizeof(size_t)));
            vector<uint32_t> visibleSparseIds(size);
            fsVisValid.read(reinterpret_cast<char*>(visibleSparseIds.data()),
                            static_cast<std::streamsize>(size * sizeof(uint32_t)));
            sparseIdsToBitset(visibleSparseIds, cellVisPoint.visibleFromCurPoint);
        }
    }
    fsVisValid.close();

    // after completing visibility, compute danger
    setDangerPoints(navFile, area);
}

void VisPoints::setDangerPoints(const nav_mesh::nav_file & navFile, bool area) {
    if (area) {
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
    else {
        // TODO: fix this when have cell connections
        /*
        for (size_t srcCell = 0; srcCell < cellVisPoints.size(); srcCell++) {
            for (size_t dangerCell = 0; dangerCell < cellVisPoints.size(); dangerCell++) {
                if (isVisibleIndex(srcCell, dangerCell)) {
                    for (size_t i = 0; i < navFile.connections_cell_length[dangerCell]; i++) {
                        size_t conCellIndex = navFile.connections[navFile.connections_cell_start[dangerCell] + i];
                        if (!isVisibleIndex(srcCell, conCellIndex)) {
                            cellVisPoints[srcCell].dangerFromCurPoint[dangerCell] = true;
                            break;
                        }
                    }
                }
            }
        }
         */
    }
}


template <size_t SZ>
vector<uint32_t> bitsetToSparseIds(const bitset<SZ> & bits) {
    vector<uint32_t> result;
    for (size_t i = 0; i < bits.size(); i++) {
        if (bits[i]) {
            result.push_back(static_cast<uint32_t>(i));
        }
    }
    return result;
}

template <size_t SZ>
void sparseIdsToBitset(const vector<uint32_t> & sparseIds, bitset<SZ> & result) {
    result = 0;
    for (const auto & id : sparseIds) {
        result[id] = true;
    }
}
