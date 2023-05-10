//
// Created by durst on 2/20/23.
//

#include "queries/nearest_nav_cell.h"
#include "file_helpers.h"
#include <filesystem>
#include <atomic>

namespace csknow::nearest_nav_cell {
    void NearestNavCell::load(const string &mapsPath, const string &mapName) {
        string nearestFileName = mapName + extension;
        string nearestFilePath = mapsPath + "/" + nearestFileName;

        if (std::filesystem::exists(nearestFilePath)) {
            std::ifstream fsNearest(nearestFilePath);
            string nearestBuf;
            size_t index = 0;
            getline(fsNearest, nearestBuf); // skip first line
            while (getline(fsNearest, nearestBuf)) {
                stringstream nearestStream(nearestBuf);
                string value;

                getline(nearestStream, value, ','); // skip index

                getline(nearestStream, value, ',');
                gridEntryAABB[index].min.x = std::stod(value);

                getline(nearestStream, value, ',');
                gridEntryAABB[index].min.y = std::stod(value);

                getline(nearestStream, value, ',');
                gridEntryAABB[index].min.z = std::stod(value);

                getline(nearestStream, value, ',');
                gridEntryAABB[index].max.x = std::stod(value);

                getline(nearestStream, value, ',');
                gridEntryAABB[index].max.y = std::stod(value);

                getline(nearestStream, value, ',');
                gridEntryAABB[index].max.z = std::stod(value);

                for (size_t i = 0; i < NUM_NEAREST_CELLS_PER_GRID_ENTRY; i++) {
                    getline(nearestStream, value, ',');
                    nearestCellsGrid[index][i].cellId = std::stoul(value);

                    getline(nearestStream, value, ',');
                    nearestCellsGrid[index][i].distance = std::stod(value);
                }
                index++;
            }
            size = static_cast<int64_t>(nearestCellsGrid.size());
        }
        else {
            throw std::runtime_error("no nearest nav cell file");
        }
    }

    void NearestNavCell::runQuery(const string & mapsPath, const string & mapName) {
        string nearestFileName = mapName + extension;
        string nearestFilePath = mapsPath + "/" + nearestFileName;

        // step 1: get bounds for all cells

        areaBounds = visPoints.getAreaBounds();

        // step 2: define grid dimensions, add 1 as size is 1 greater than max 0 indexed coordinate
        gridDimensions = posToGridIndex(areaBounds.max) + 1;

        // step 3: for every entry in grid, find closest in map (loading this if possible)
        gridEntryAABB.resize(gridDimensions.x * gridDimensions.y * gridDimensions.z);
        nearestCellsGrid.resize(gridDimensions.x * gridDimensions.y * gridDimensions.z);
        if (std::filesystem::exists(nearestFilePath)) {
            load(mapsPath, mapName);
        }
        else {
            std::atomic<int64_t> xProcessed = 0;
#pragma omp parallel for
            for (int64_t curXId = 0; curXId < gridDimensions.x; curXId++) {
                for (int64_t curYId = 0; curYId < gridDimensions.y; curYId++) {
                    for (int64_t curZId = 0; curZId < gridDimensions.z; curZId++) {
                        IVec3 gridIndex = {curXId, curYId, curZId};

                        gridIndexToAABB(gridIndex) = {gridIndexToMinPos(gridIndex), gridIndexToMinPos(gridIndex + 1)};

                        vector<CellIdAndDistance> cellVisPointsByDistance =
                            visPoints.getCellVisPointsByDistance(gridIndexToCenterPos(gridIndex),
                                                                 NUM_NEAREST_CELLS_PER_GRID_ENTRY,
                                                                 NUM_NEAREST_CELLS_PER_GRID_ENTRY);
                        for (size_t i = 0; i < NUM_NEAREST_CELLS_PER_GRID_ENTRY; i++) {
                            gridIndexToNearestCells(gridIndex)[i] = cellVisPointsByDistance[i];
                        }
                    }
                }
                xProcessed++;
                printProgress(xProcessed, gridDimensions.x);
            }

            size = static_cast<int64_t>(nearestCellsGrid.size());
            save(mapsPath, mapName);
        }
    }

    std::vector<CellIdAndDistance> NearestNavCell::getNearestCells(Vec3 pos) const {
        IVec3 curGridIndex = posToGridIndex(pos);
        const NearestGridData & nearestGridData = gridIndexToNearestCells(curGridIndex);
        // get the nearest grid index other than the cur one
        // take nearest in x/y with same z
        Vec3 curGridCenter = gridIndexToCenterPos(curGridIndex);
        IVec3 otherGridIndex = curGridIndex;
        otherGridIndex.x += pos.x >= curGridCenter.x ? 1 : -1;
        // if on edge, mirror reflect so not out of bounds
        if (otherGridIndex.x >= gridDimensions.x) {
            otherGridIndex.x -= 2;
        }
        else if (otherGridIndex.x < 0) {
            otherGridIndex.x = 1;
        }
        otherGridIndex.y += pos.y >= curGridCenter.y ? 1 : -1;
        if (otherGridIndex.y >= gridDimensions.y) {
            otherGridIndex.y -= 2;
        }
        else if (otherGridIndex.y < 0) {
            otherGridIndex.y = 1;
        }
        const NearestGridData & otherGridData = gridIndexToNearestCells(otherGridIndex);

        CellIdAndDistance firstNearest = nearestGridData[0];
        firstNearest.distance = vis_point_helpers::get_point_to_aabb_distance(
              pos, visPoints.getCellVisPoints()[firstNearest.cellId].cellCoordinates);
        if (firstNearest.distance == 0.) {
            CellIdAndDistance secondNearest = otherGridData[0];
            secondNearest.distance = vis_point_helpers::get_point_to_aabb_distance(
                pos, visPoints.getCellVisPoints()[firstNearest.cellId].cellCoordinates);
            return {firstNearest, secondNearest};
        }
        else {
            std::set<CellId> resultSet;
            std::vector<CellIdAndDistance> result;
            for (const auto & gridData : {nearestGridData, otherGridData}) {
                for (const auto & nearestGridEntry : gridData) {
                    if (resultSet.find(nearestGridEntry.cellId) == resultSet.end()) {
                        resultSet.insert(nearestGridEntry.cellId);
                        result.push_back({nearestGridEntry.cellId, vis_point_helpers::get_point_to_aabb_distance(
                            pos, visPoints.getCellVisPoints()[nearestGridEntry.cellId].cellCoordinates)});
                    }
                }

            }

            std::sort(result.begin(), result.end(), [](const CellIdAndDistance & a, const CellIdAndDistance & b) {
                return a.distance < b.distance;
            });
            return result;
        }
    }

    AreaId NearestNavCell::getNearestArea(Vec3 pos) const {
        IVec3 curGridIndex = posToGridIndex(pos);
        const NearestGridData & nearestGridData = gridIndexToNearestCells(curGridIndex);
        // get the nearest grid index other than the cur one
        // take nearest in x/y with same z
        Vec3 curGridCenter = gridIndexToCenterPos(curGridIndex);
        IVec3 otherGridIndex = curGridIndex;
        otherGridIndex.x += pos.x >= curGridCenter.x ? 1 : -1;
        otherGridIndex.y += pos.y >= curGridCenter.y ? 1 : -1;
        const NearestGridData & otherGridData = gridIndexToNearestCells(otherGridIndex);

        std::set<AreaId> resultSet;
        std::vector<AreaIdAndDistance> result;
        for (const auto & gridData : {nearestGridData, otherGridData}) {
            for (const auto & nearestGridEntry : gridData) {
                AreaId areaId = visPoints.getCellVisPoints()[nearestGridEntry.cellId].areaId;
                if (resultSet.find(areaId) == resultSet.end()) {
                    resultSet.insert(areaId);
                    result.push_back({areaId, vis_point_helpers::get_point_to_area_aabb_distance(
                        pos, visPoints.getAreaVisPoint(areaId).areaCoordinates)
                    });
                }
            }

        }

        std::sort(result.begin(), result.end(), [](const AreaIdAndDistance & a, const AreaIdAndDistance & b) {
            return a.distance < b.distance;
        });

        return result[0].areaId;
    }
}

