//
// Created by steam on 7/4/22.
//

#ifndef CSKNOW_MEMORY_DATA_H
#define CSKNOW_MEMORY_DATA_H

#include "load_save_bot_data.h"
#include "navmesh/nav_file.h"
using std::map;

struct EnemyPositionMemory {
    Vec3 lastSeenFootPos;
    int32_t lastSeenFrame;
    //set<uint32_t> believedAreas;
    //map<uint32_t, int32_t> frameCouldEnterArea;
};

struct EnemyPositionsMemory {
    // if this is for all teammates, then use players on right team, otherwise just use source player
    bool considerAllTeammates;
    int32_t team;
    CSGOId srcPlayer;
    map<CSGOId, EnemyPositionMemory> positions;

    void updatePositions(const ServerState & state, nav_mesh::nav_file & navFile, double maxMemorySeconds);
};

#endif //CSKNOW_MEMORY_DATA_H
