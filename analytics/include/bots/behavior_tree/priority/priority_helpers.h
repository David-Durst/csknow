//
// Created by durst on 5/4/22.
//

#ifndef CSKNOW_PRIORITY_HELPERS_H
#define CSKNOW_PRIORITY_HELPERS_H

#include "bots/behavior_tree/node.h"

void moveToWaypoint(const Blackboard & blackboard, const ServerState & state, TreeThinker & treeThinker,
                    const Order & curOrder, Priority & curPriority);

int64_t getMaxFinishedWaypoint(const Blackboard & blackboard, const ServerState & state,
                               const Order & curOrder, Priority & curPriority,
                               CSGOId playerId, string curPlace, AreaId curAreaId);

#endif //CSKNOW_PRIORITY_HELPERS_H
