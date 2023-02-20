//
// Created by durst on 2/20/23.
//

#include "queries/nearest_nav_area.h"

namespace csknow::nearest_nav_area {
    void NearestNavArea::runQuery(const Rounds & rounds, const Ticks & ticks, const PlayerAtTick & playerAtTick,
                                  const nav_mesh::nav_file & navFile) {
        playerPositionBounds = {{
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()
            }, {
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
        nearestAreasGrid.resize(gridDimensions.x * gridDimensions.y * gridDimensions.z);
        for (int64_t curXId = 0; curXId < gridDimensions.x; curXId++) {
            for (int64_t curYId = 0; curYId < gridDimensions.y; curYId++) {
                for (int64_t curZId = 0; curZId < gridDimensions.z; curZId++) {
                    IVec3 gridIndex = {curXId, curYId, curZId};
                    vector<nav_mesh::AreaDistance> areaDistances =
                        navFile.get_area_distances_to_position(vec3Conv(gridIndexToPos(gridIndex)));
                    gridIndexToNearestAreas(gridIndex)[0] = areaDistances[0];
                    gridIndexToNearestAreas(gridIndex)[1] = areaDistances[1];
                }
            }
        }

        size = nearestAreasGrid.size();
    }
}