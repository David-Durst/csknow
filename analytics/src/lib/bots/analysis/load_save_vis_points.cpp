//
// Created by steam on 7/9/22.
//

#include "bots/analysis/load_save_vis_points.h"
#include <filesystem>

void VisPoints::createAreaVisPoints(const nav_mesh::nav_file & navFile) {
    for (const auto & navArea : navFile.m_areas) {
        areaVisPoints.push_back(AreaVisPoint{navArea.get_id(), navArea.m_place,
                                             {vec3tConv(navArea.get_min_corner()), vec3tConv(navArea.get_max_corner())},
                                             vec3tConv(navArea.get_center())});
        areaVisPoints.back().center.z += EYE_HEIGHT;
        placeIdToAreaIds[navArea.m_place].push_back(navArea.get_id());
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
    maxCellNumbersByDim = {
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
    for (auto & areaVisPoint : areaVisPoints) {
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
        for (int64_t curXId = minCellMinInArea.x; curXId <= maxCellMinInArea.x && curXId < maxCellNumbersByDim[0];
             curXId++) {
            for (int64_t curYId = minCellMinInArea.y; curYId <= maxCellMinInArea.y && curYId < maxCellNumbersByDim[1];
                 curYId++) {
                for (int64_t curZId = minCellMinInArea.z; curZId <= maxCellMinInArea.z && curZId < maxCellNumbersByDim[2];
                     curZId++) {
                    Vec3 cellMin = areaBounds.min +
                        Vec3{CELL_DIM_WIDTH_DEPTH * static_cast<double>(curXId),
                             CELL_DIM_WIDTH_DEPTH * static_cast<double>(curYId),
                             CELL_DIM_HEIGHT * static_cast<double>(curZId)};
                    Vec3 cellMax = cellMin + Vec3{CELL_DIM_WIDTH_DEPTH, CELL_DIM_WIDTH_DEPTH, CELL_DIM_HEIGHT};
                    Vec3 cellCenter = (cellMin + cellMax) / 2.;
                    Vec3 cellTopCenter = cellCenter;
                    cellTopCenter.z = cellMax.z;
                    if (pointInRegionMaxInclusive(extendedAABB, cellCenter)) {
                        areaVisPoint.cells.push_back(static_cast<CellId>(cellVisPoints.size()));
                        cellVisPoints.push_back({
                            areaVisPoint.areaId,
                            static_cast<CellId>(cellVisPoints.size()),
                            {curXId, curYId, curZId},
                            {cellMin, cellMax},
                            cellCenter,
                            cellTopCenter,
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
            for (int64_t curZId = 0; curZId < maxCellNumbersByDim[2]; curZId++) {
                int64_t linearCellAddress = curXId * maxCellNumbersByDim[1] * maxCellNumbersByDim[2] +
                                            curYId * maxCellNumbersByDim[2] + curZId;
                if (numAreasContainingCell[linearCellAddress] == 1 &&
                    singleWideAreaContainingCell[linearCellAddress] != INVALID_ID) {
                    AreaVisPoint & areaVisPoint =
                        areaVisPoints[areaIdToVectorIndex[
                            static_cast<AreaId>(singleWideAreaContainingCell[linearCellAddress])]];
                    Vec3 cellMin = areaBounds.min +
                                   Vec3{CELL_DIM_WIDTH_DEPTH * static_cast<double>(curXId),
                                        CELL_DIM_WIDTH_DEPTH * static_cast<double>(curYId),
                                        CELL_DIM_HEIGHT * static_cast<double>(curZId)};
                    Vec3 cellMax = cellMin + Vec3{CELL_DIM_WIDTH_DEPTH, CELL_DIM_WIDTH_DEPTH, CELL_DIM_HEIGHT};
                    Vec3 cellCenter = (cellMin + cellMax) / 2.;
                    Vec3 cellTopCenter = cellCenter;
                    cellTopCenter.z = cellMax.z;
                    areaVisPoint.cells.push_back(static_cast<CellId>(cellVisPoints.size()));
                    cellVisPoints.push_back({
                        areaVisPoint.areaId,
                        static_cast<CellId>(cellVisPoints.size()),
                        {curXId, curYId, curZId},
                        {cellMin, cellMax},
                        cellCenter,
                        cellTopCenter,
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
                visPointsStream << visPoint.topCenter.toCSV() << std::endl;
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
                areaVisPoints[visPoints[0]].visibleFromCurPoint.set(visPoints[1], true);
            }
            else {
                cellVisPoints[visPoints[0]].visibleFromCurPoint.set(visPoints[1], true);
            }
        }
        return true;
    }
    else {
        return false;
    }
}

void VisPoints::save(const string & mapsPath, const string & mapName, bool area) {
    string visValidFileName = getVisFileName(mapName, area, false);
    string visValidFilePath = mapsPath + "/" + visValidFileName;

    size_t visBytesSize;
    if (area) {
        visBytesSize = areaVisPoints.front().visibleFromCurPoint.internalLength() * areaVisPoints.size();
    }
    else {
        visBytesSize = cellVisPoints.front().visibleFromCurPoint.internalLength() * cellVisPoints.size();

    }
    vector<std::uint8_t> visBytes;
    visBytes.reserve(visBytesSize);
    if (area) {
        for (auto & areaVisPoint : areaVisPoints) {
            areaVisPoint.visibleFromCurPoint.exportSlice(visBytes);
        }
    }
    else {
        for (auto & cellVisPoint : cellVisPoints) {
            cellVisPoint.visibleFromCurPoint.exportSlice(visBytes);
        }
    }

    std::ofstream fsVisValid(visValidFilePath, std::ios::out | std::ios::binary);
    fsVisValid.write(reinterpret_cast<const char*>(visBytes.data()), static_cast<std::streamsize>(visBytes.size()));
    fsVisValid.close();

    string gzipCommand = "gzip -f " + visValidFilePath;
    if (std::system(gzipCommand.c_str()) != 0) {
        std::cerr << "save gzip failed" << std::endl;
    }
}

void VisPoints::load(const string & mapsPath, const string & mapName, bool area, const nav_mesh::nav_file & navFile,
                     bool fixSymmetry) {
    string visValidFileName = getVisFileName(mapName, area, false);
    string visValidFilePath = mapsPath + "/" + visValidFileName;

    string gzipCommand = "gzip -dfk " + visValidFilePath + ".gz";
    if (std::system(gzipCommand.c_str()) != 0) {
        std::cerr << "load gzip failed" << std::endl;
    }

    size_t visBytesSize;
    if (area) {
        visBytesSize = areaVisPoints.front().visibleFromCurPoint.internalLength() * areaVisPoints.size();
    }
    else {
        visBytesSize = cellVisPoints.front().visibleFromCurPoint.internalLength() * cellVisPoints.size();

    }
    vector<std::uint8_t> visBytes(visBytesSize, 0);

    std::ifstream fsVisValid(visValidFilePath, std::ios::in | std::ios::binary);
    fsVisValid.read(reinterpret_cast<char*>(visBytes.data()), static_cast<std::streamsize>(visBytes.size()));
    size_t visBytesOffset = 0;
    if (area) {
        for (auto & areaVisPoint : areaVisPoints) {
            areaVisPoint.visibleFromCurPoint.assignSlice(visBytes, visBytesOffset);
            visBytesOffset += areaVisPoint.visibleFromCurPoint.getInternal().size();
        }
    }
    else {
        for (auto & cellVisPoint : cellVisPoints) {
            cellVisPoint.visibleFromCurPoint.assignSlice(visBytes, visBytesOffset);
            visBytesOffset += cellVisPoint.visibleFromCurPoint.getInternal().size();
        }
    }

    if (fixSymmetry) {
        fix_symmetry(area);
    }

    // after completing visibility, compute danger
    setDangerPoints(navFile, area);
}

void VisPoints::fix_symmetry(bool area) {
    if (area) {
        for (size_t i = 0; i < areaVisPoints.size(); i++) {
            for (size_t j = 0; j <= i; j++) {
                areaVisPoints[i].visibleFromCurPoint.set(j, i == j ||
                    areaVisPoints[i].visibleFromCurPoint[j] || areaVisPoints[j].visibleFromCurPoint[i]);
            }
        }
    }
    else {
#pragma omp parallel for
        for (size_t i = 0; i < cellVisPoints.size(); i++) {
            for (size_t j = 0; j <= i; j++) {
                cellVisPoints[i].visibleFromCurPoint.set(j, i == j ||
                    cellVisPoints[i].visibleFromCurPoint[j] || cellVisPoints[j].visibleFromCurPoint[i]);
            }
        }
    }
}


void VisPoints::setDangerPoints(const nav_mesh::nav_file & navFile, bool area) {
    if (area) {
        for (size_t srcArea = 0; srcArea < areaVisPoints.size(); srcArea++) {
            for (size_t dangerArea = 0; dangerArea < areaVisPoints.size(); dangerArea++) {
                if (isVisibleIndex(srcArea, dangerArea)) {
                    for (size_t i = 0; i < navFile.connections_area_length[dangerArea]; i++) {
                        size_t conAreaIndex = navFile.connections[navFile.connections_area_start[dangerArea] + i];
                        if (!isVisibleIndex(srcArea, conAreaIndex)) {
                            areaVisPoints[srcArea].dangerFromCurPoint.set(dangerArea, true);
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

namespace vis_point_helpers {
// https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
    bool is_within_3d( const AABB & area, const Vec3 & position, float z_tolerance) {
        if ( position.x < area.min.x )
            return false;

        if ( position.x > area.max.x )
            return false;

        if ( position.y < area.min.y )
            return false;

        if ( position.y > area.max.y )
            return false;

        if ( position.z < area.min.z - z_tolerance )
            return false;

        if ( position.z > area.max.z + z_tolerance )
            return false;

        return true;
    }

    float get_point_to_aabb_distance( Vec3 position, const AABB& area) {
        if (is_within_3d(area, position, 0.)) {
            return 0.;
        }
        float dx = std::max(area.min.x - position.x, std::max(0., position.x - area.max.x));
        float dy = std::max(area.min.y - position.y, std::max(0., position.y - area.max.y));
        float dz = std::max(area.min.z - position.z, std::max(0., position.z - area.max.z));
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    float get_point_to_area_aabb_distance( Vec3 position, const AABB& area) {
        // see is_within_3d in nav_area.h for why 31
        if (is_within_3d(area, position, 31.)) {
            return 0.;
        }
        float dx = std::max(area.min.x - position.x, std::max(0., position.x - area.max.x));
        float dy = std::max(area.min.y - position.y, std::max(0., position.y - area.max.y));
        float dz = std::max(area.min.z - position.z, std::max(0., position.z - area.max.z));
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
}

vector<CellIdAndDistance> VisPoints::getCellVisPointsByDistance(const Vec3 &pos, size_t maxAreasToConsider,
                                                                size_t maxCellsToConsider) const {
    vector<nav_mesh::AreaDistance> areaDistances = navFile.get_area_distances_to_position(vec3Conv(pos));
    // a small number of areas have no cells, pick at least first two area that's closest and enough that at least
    // three cells
    // this ensures having coverage on an area boundary and being between two cells that aren't from different
    // crouching heights
    size_t numAreasToConsider = 0;
    size_t numCellsToConsider = 0;
    for (const auto & areaDistance : areaDistances) {
        numAreasToConsider++;
        numCellsToConsider += areaVisPoints[areaIdToIndex(areaDistance.areaId)].cells.size();
        if (numAreasToConsider >= maxAreasToConsider && numCellsToConsider >= maxCellsToConsider) {
            break;
        }
    }

    vector<CellIdAndDistance> result;
    for (size_t i = 0; i < numAreasToConsider; i++) {
        const AreaVisPoint & areaVisPoint = areaVisPoints[areaIdToVectorIndex.at(areaDistances[i].areaId)];
        for (size_t j = 0; j < areaVisPoint.cells.size(); j++) {
            const auto & cellId = areaVisPoint.cells[j];
            result.push_back({
                cellId,
                vis_point_helpers::get_point_to_aabb_distance(pos, cellVisPoints[cellId].cellCoordinates)
            });
        }
    }
    std::sort(result.begin(), result.end(), [](const CellIdAndDistance & a, const CellIdAndDistance & b) {
        return a.distance < b.distance;
    });
    return result;
}
