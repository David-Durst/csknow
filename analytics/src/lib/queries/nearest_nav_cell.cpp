//
// Created by durst on 2/20/23.
//

#include "queries/nearest_nav_cell.h"
#include "file_helpers.h"
#include <filesystem>
#include <atomic>

namespace csknow::nearest_nav_area {
    void NearestNavCell::load(const string &mapsPath, const string &mapName) {
        string nearestFileName = mapName + ".near";
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

                getline(nearestStream, value, ',');
                nearestCellsGrid.push_back({});
                nearestCellsGrid[index][0].cellId = std::stoul(value);

                getline(nearestStream, value, ',');
                nearestCellsGrid[index][0].distance = std::stod(value);

                nearestCellsGrid.push_back({});
                nearestCellsGrid[index][1].cellId = std::stoul(value);

                getline(nearestStream, value, ',');
                nearestCellsGrid[index][1].distance = std::stod(value);

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

    void NearestNavCell::runQuery(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                  const VisPoints & visPoints, const string & mapsPath, const string & mapName) {
        string nearestFileName = mapName + ".near";
        string nearestFilePath = mapsPath + "/" + nearestFileName;

        if (std::filesystem::exists(nearestFilePath)) {
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

            // step 1: get bounds for where players can stand in data
            for (int64_t roundIndex = 0; roundIndex < rounds.size; roundIndex++) {
                for (int64_t tickIndex = rounds.ticksPerRound[roundIndex].minId;
                     tickIndex <= rounds.ticksPerRound[roundIndex].maxId; tickIndex++) {
                    for (int64_t patIndex = ticks.patPerTick[tickIndex].minId;
                         patIndex <= ticks.patPerTick[tickIndex].maxId; patIndex++) {
                        playerPositionBounds.min = min(playerPositionBounds.min, {
                            playerAtTick.posX[patIndex],
                            playerAtTick.posY[patIndex],
                            playerAtTick.posZ[patIndex],
                        });
                        playerPositionBounds.max = max(playerPositionBounds.max, {
                            playerAtTick.posX[patIndex],
                            playerAtTick.posY[patIndex],
                            playerAtTick.posZ[patIndex],
                        });

                    }
                }
            }

            // step 2: define grid dimensions, add 1 as size is 1 greater than max 0 indexed coordinate
            gridDimensions = posToGridIndex(playerPositionBounds.max) + 1;

            // step 3: for every entry in grid, find closest in map
            gridEntryAABB.resize(gridDimensions.x * gridDimensions.y * gridDimensions.z);
            nearestCellsGrid.resize(gridDimensions.x * gridDimensions.y * gridDimensions.z);
            std::atomic<int64_t> xProcessed = 0;
            std::cout << "grid dimensions (" << gridDimensions.x << "," << gridDimensions.y << "," << gridDimensions.z << ")" << std::endl;
#pragma omp parallel for
            for (int64_t curXId = 0; curXId < gridDimensions.x; curXId++) {
                for (int64_t curYId = 0; curYId < gridDimensions.y; curYId++) {
                    for (int64_t curZId = 0; curZId < gridDimensions.z; curZId++) {
                        IVec3 gridIndex = {curXId, curYId, curZId};

                        gridIndexToAABB(gridIndex) = {gridIndexToPos(gridIndex), gridIndexToPos(gridIndex) + 1};

                        vector<CellIdAndDistance> cellVisPointsByDistance =
                            visPoints.getCellVisPointsByDistance(gridIndexToPos(gridIndex));
                        gridIndexToNearestCells(gridIndex)[0] = cellVisPointsByDistance[0];
                        gridIndexToNearestCells(gridIndex)[1] = cellVisPointsByDistance[1];
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