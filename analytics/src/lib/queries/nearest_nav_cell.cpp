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
                gridEntryAABB.push_back({});
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

                nearestCellsGrid.push_back({});
                for (size_t i = 0; i < NUM_NEAREST_CELLS_PER_GRID_ENTRY; i++) {
                    getline(nearestStream, value, ',');
                    nearestCellsGrid[index][i].cellId = std::stoul(value);

                    getline(nearestStream, value, ',');
                    nearestCellsGrid[index][i].distance = std::stod(value);
                }
                index++;
            }
            size = static_cast<int64_t>(nearestCellsGrid.size());
            playerPositionBounds.min = gridEntryAABB.front().min;
            playerPositionBounds.max = gridEntryAABB.back().max;
            gridDimensions = posToGridIndex(gridEntryAABB.back().min) + 1;

        }
        else {
            throw std::runtime_error("no nearest nav cell file");
        }
    }

    void NearestNavCell::runQuery(const string & mapsPath, const string & mapName) {
        string nearestFileName = mapName + extension;
        string nearestFilePath = mapsPath + "/" + nearestFileName;

        if (false && std::filesystem::exists(nearestFilePath)) {
            load(mapsPath, mapName);
        }
        else {
            playerPositionBounds = {{
                                        std::numeric_limits<double>::max(),
                                        std::numeric_limits<double>::max(),
                                        std::numeric_limits<double>::max()
                                    },
                                    {
                                        -1. * std::numeric_limits<double>::max(),
                                        -1. * std::numeric_limits<double>::max(),
                                        -1. * std::numeric_limits<double>::max()
                                    }};

            // step 1: get bounds for all cells
            for (const auto & cellVisPoint : visPoints.getCellVisPoints()) {
                playerPositionBounds.min = min(playerPositionBounds.min, cellVisPoint.cellCoordinates.min);
                playerPositionBounds.max = max(playerPositionBounds.max, cellVisPoint.cellCoordinates.max);
            }

            // step 2: define grid dimensions, add 1 as size is 1 greater than max 0 indexed coordinate
            gridDimensions = posToGridIndex(playerPositionBounds.max) + 1;

            // step 3: for every entry in grid, find closest in map
            gridEntryAABB.resize(gridDimensions.x * gridDimensions.y * gridDimensions.z);
            nearestCellsGrid.resize(gridDimensions.x * gridDimensions.y * gridDimensions.z);
            std::atomic<int64_t> xProcessed = 0;
#pragma omp parallel for
            for (int64_t curXId = 0; curXId < gridDimensions.x; curXId++) {
                for (int64_t curYId = 0; curYId < gridDimensions.y; curYId++) {
                    for (int64_t curZId = 0; curZId < gridDimensions.z; curZId++) {
                        IVec3 gridIndex = {curXId, curYId, curZId};

                        gridIndexToAABB(gridIndex) = {gridIndexToCenterPos(gridIndex), gridIndexToCenterPos(gridIndex + 1)};

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

            size = nearestCellsGrid.size();
            save(mapsPath, mapName);
        }
    }
}