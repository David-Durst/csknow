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

    Vec3 deltaAreaBounds = areaBounds.max - areaBounds.min;
    array<int64_t, 3> maxCellNumbersByDim{
        static_cast<int64_t>(deltaAreaBounds.x / CELL_DIM_WIDTH_DEPTH),
        static_cast<int64_t>(deltaAreaBounds.y / CELL_DIM_WIDTH_DEPTH),
        static_cast<int64_t>(deltaAreaBounds.z / CELL_DIM_HEIGHT)
    };

    // track valid cells that are in exactly one cell and that cell is single wide but aren't assigned
    // this filters out edge cases of cells near walls, but keeps ledges
    vector<int8_t> numAreasContainingCell(
        maxCellNumbersByDim[0] * maxCellNumbersByDim[1] * maxCellNumbersByDim[2], 0);
    vector<int64_t> singleWideAreaContainingCell(
        maxCellNumbersByDim[0] * maxCellNumbersByDim[1] * maxCellNumbersByDim[2], INVALID_ID);

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
                    else {
                        int64_t linearCellAddress = curXId * maxCellNumbersByDim[1] * maxCellNumbersByDim[2] +
                                                    curYId * maxCellNumbersByDim[2] + curZId;
                        numAreasContainingCell[linearCellAddress]++;
                        // every area is at least half player width, so every area should be at least size 1
                        // removing double hits from both start and end
                        if ((maxCellMinInArea.x - minCellMinInArea.x == 1 && curXId == minCellMinInArea.x) ||
                            (maxCellMinInArea.y - minCellMinInArea.y == 1 && curYId == minCellMinInArea.y) ||
                            (maxCellMinInArea.z - minCellMinInArea.z == 1 && curZId == minCellMinInArea.z)) {
                            singleWideAreaContainingCell[linearCellAddress] = static_cast<int64_t>(areaVisPoint.areaId);
                        }
                    }
                }
            }
        }

    }

    // add missing cells
    for (int64_t curXId = 0; curXId < maxCellNumbersByDim[0]; curXId++) {
        for (int64_t curYId = 0; curYId < maxCellNumbersByDim[1]; curYId++) {
            for (int64_t curZId = 0; curZId <= maxCellNumbersByDim[2]; curZId++) {
                int64_t linearCellAddress = curXId * maxCellNumbersByDim[1] * maxCellNumbersByDim[2] +
                                            curYId * maxCellNumbersByDim[2] + curZId;
                if (numAreasContainingCell[linearCellAddress] == 1 &&
                    singleWideAreaContainingCell[linearCellAddress] != INVALID_ID) {
                    const AreaVisPoint & areaVisPoint =
                        areaVisPoints[areaIdToVectorIndex[
                            static_cast<AreaId>(singleWideAreaContainingCell[linearCellAddress])]];
                    Vec3 cellMin = areaBounds.min +
                                   Vec3{CELL_DIM_WIDTH_DEPTH * static_cast<double>(curXId),
                                        CELL_DIM_WIDTH_DEPTH * static_cast<double>(curYId),
                                        CELL_DIM_HEIGHT * static_cast<double>(curZId)};
                    Vec3 cellMax = cellMin + Vec3{CELL_DIM_WIDTH_DEPTH, CELL_DIM_WIDTH_DEPTH, CELL_DIM_HEIGHT};
                    Vec3 cellCenter = (cellMin + cellMax) / 2.;
                    cellVisPoints.push_back({
                        areaVisPoint.areaId,
                        static_cast<CellId>(cellVisPoints.size()),
                        {cellMin, cellMax},
                        cellCenter
                    });
                }
            }
        }
    }

#if false
    // checks that nav cells are non-overlapping
    for (size_t i = 0; i < cellVisPoints.size(); i++) {
        for (size_t j = i+1; j < cellVisPoints.size(); j++) {
            // https://www.youtube.com/watch?v=tBoRp5K8HU0
            if (cellVisPoints[i].areaId == 8251 || cellVisPoints[j].areaId == 8251) {
                continue;
            }
            if (aabbOverlapExclusive(cellVisPoints[i].cellCoordinates, cellVisPoints[j].cellCoordinates)) {
                size_t areaIndexI = areaIdToVectorIndex[cellVisPoints[i].areaId];
                size_t areaIndexJ = areaIdToVectorIndex[cellVisPoints[j].areaId];
                AreaVisPoint ai = areaVisPoints[areaIndexI];
                AreaVisPoint aj = areaVisPoints[areaIndexJ];
                std::cout << "found overlapping nav cells" << std::endl;
            }
        }
    }
#endif // false

}

void VisPoints::clearFiles(const ServerState & state) {
    string visValidFileName = "vis_valid.csv";
    string visValidFilePath = state.dataPath + "/" + visValidFileName;
    string tmpVisValidFileName = "vis_valid.csv.tmp.read";
    string tmpVisValidFilePath = state.dataPath + "/" + tmpVisValidFileName;

    std::filesystem::remove(visValidFilePath);
    std::filesystem::remove(tmpVisValidFilePath);
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
    string tmpVisValidFileName = "vis_valid.csv.tmp.read";
    string tmpVisValidFilePath = state.dataPath + "/" + tmpVisValidFileName;

    if (std::filesystem::exists(visValidFilePath)) {
        std::filesystem::rename(visValidFilePath, tmpVisValidFilePath);

        std::ifstream fsVisValid(tmpVisValidFilePath);

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

    std::stringstream validStream;

    if (area) {
        for (const auto & areaVisPoint : areaVisPoints) {
            validStream << bitsetToBase64(areaVisPoint.visibleFromCurPoint) << std::endl;
        }
    }
    else {
        for (const auto & cellVisPoint : cellVisPoints) {
            validStream << bitsetToBase64(cellVisPoint.visibleFromCurPoint) << std::endl;
        }
    }

    std::ofstream fsVisValid(visValidFilePath, std::ios::out);
    fsVisValid << validStream.str();
    fsVisValid.close();
}

void VisPoints::new_load(const string & mapsPath, const string & mapName, bool area, const nav_mesh::nav_file & navFile) {
    string visValidFileName = mapName + (area ? "_area" : "_cell") + ".vis";
    string visValidFilePath = mapsPath + "/" + visValidFileName;

    std::ifstream fsVisValid(visValidFilePath, std::ios::in | std::ios::binary);
    string visBuf;
    if (area) {
        for (auto & areaVisPoint : areaVisPoints) {
            getline(fsVisValid, visBuf); // skip first line
            base64ToBitset(visBuf, areaVisPoint.visibleFromCurPoint);
        }
    }
    else {
        for (auto & cellVisPoint : cellVisPoints) {
            getline(fsVisValid, visBuf); // skip first line
            base64ToBitset(visBuf, cellVisPoint.visibleFromCurPoint);
        }
    }
    fsVisValid.close();

    // after completing visibility, compute danger
    setDangerPoints(navFile, area);
}

void VisPoints::load(const string & mapsPath, const string & mapName, bool area, const nav_mesh::nav_file & navFile) {
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
    setDangerPoints(navFile, true);
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
string bitsetToBase64(const bitset<SZ> & bits) {
    bitset<SZ> firstByteMask(255);
    vector<base64::byte> result;
    for (size_t i = 0; i < bits.size() / 8; i += 8) {
        bitset<SZ> masked((bits << i) & firstByteMask);
        result.push_back(static_cast<base64::byte>(masked.to_ulong()));
    }
    return base64::encode(result);
}

template <size_t SZ>
void base64ToBitset(const string & base64Input, bitset<SZ> & bits) {
    vector<base64::byte> input = base64::decode(base64Input);
    for (int64_t i = static_cast<int64_t>(input.size()) - 1; i >= 0; i--) {
        bitset<SZ> curVal(input[i]);
        bits |= curVal;
        if (i != 0) {
            bits >>= 8;
        }
    }
}
