//
// Created by steam on 7/31/22.
//

#ifndef CSKNOW_SPACING_HELPERS_H
#define CSKNOW_SPACING_HELPERS_H
#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/pathing_node.h"
#define MIN_BAIT_DISTANCE 50.f
#define MAX_BAIT_DISTANCE 200.f

struct NumAheadResult {
    int numAhead;
    double nearestInFront;
};

NumAheadResult computeNumAhead(Blackboard & blackboard, const ServerState & state, const ServerState::Client & curClient);

#endif //CSKNOW_SPACING_HELPERS_H
