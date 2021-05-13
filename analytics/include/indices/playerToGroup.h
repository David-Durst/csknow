#ifndef CSKNOW_PLAYERTOGROUP_H
#define CSKNOW_PLAYERTOGROUP_H
#include "load_data.h"
#include "queries/grouping.h"
#include "omp.h"

class PlayerToGroupIndex {
public:
    // for each player, for each tick, a vector of indices in grouping
    vector<vector<int64_t>> groupsPerPlayerTick[NUM_PLAYERS];
    PlayerToGroupIndex(const Ticks & ticks, const PlayerAtTick & pat, const GroupingResult & groupingResult);
};

#endif //CSKNOW_PLAYERTOGROUP_H
