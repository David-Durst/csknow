//
// Created by durst on 4/21/23.
//

#include "bots/load_save_bot_data.h"
#include <map>

#ifndef CSKNOW_ORDER_MODEL_HEURISTIC_MAPPING_H
#define CSKNOW_ORDER_MODEL_HEURISTIC_MAPPING_H

struct OrderId {
    TeamId team = INVALID_ID;
    int64_t index = INVALID_ID;
};

static bool operator<(const OrderId& a, const OrderId& b) {
    return a.team < b.team || (a.team == b.team && a.index < b.index);
}

const std::map<size_t, size_t> aHeuristicToModelOrderIndices{
    {1, 0}, // a spawn
    {0, 1}, // a long
    {2, 2}, // a cat
};
const std::map<size_t, size_t> bHeuristicToModelOrderIndices{
    {1, 3}, // b hole
    {0, 4}, // b doors
    {2, 5} // b tuns
};

#endif //CSKNOW_ORDER_MODEL_HEURISTIC_MAPPING_H
