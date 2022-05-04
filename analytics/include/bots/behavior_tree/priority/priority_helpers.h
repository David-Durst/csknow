//
// Created by durst on 5/4/22.
//

#ifndef CSKNOW_PRIORITY_HELPERS_H
#define CSKNOW_PRIORITY_HELPERS_H

#include "bots/behavior_tree/node.h"

void moveToWaypoint(Node & node, const ServerState & state, TreeThinker & treeThinker,
                    const Order & curOrder, Priority & curPriority);

void finishWaypoint(Node & node, const ServerState & state, TreeThinker & treeThinker,
                    const Order & curOrder, Priority & curPriority, string curPlace);

#endif //CSKNOW_PRIORITY_HELPERS_H
