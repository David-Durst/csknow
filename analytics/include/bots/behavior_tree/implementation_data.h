//
// Created by durst on 5/8/22.
//

#ifndef CSKNOW_IMPLEMENTATION_DATA_H
#define CSKNOW_IMPLEMENTATION_DATA_H
#include "load_save_bot_data.h"
#include "bots/behavior_tree/priority/priority_data.h"

struct Path {
    bool pathCallSucceeded;
    vector<Vec3> waypoints;
    uint32_t pathEndAreaId;
};

#endif //CSKNOW_IMPLEMENTATION_DATA_H
