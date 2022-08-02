//
// Created by steam on 7/31/22.
//

#ifndef CSKNOW_SPACING_HELPERS_H
#define CSKNOW_SPACING_HELPERS_H
#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/pathing_node.h"
#define MIN_BAIT_DISTANCE 200.f
//#define START_BAIT_DISTANCE 200.f
#define MAX_PUSH_DISTANCE 1500.f
#define MAX_BAIT_DISTANCE 2000.f

struct NumAheadResult {
    int numAhead;
    int numBehind;
    double nearestInFront;
    double nearestBehind;
};

NumAheadResult computeNumAhead(Blackboard & blackboard, const ServerState & state, const ServerState::Client & curClient);

#endif //CSKNOW_SPACING_HELPERS_H
