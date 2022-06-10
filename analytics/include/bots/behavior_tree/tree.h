//
// Created by durst on 5/9/22.
//

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/order_node.h"
#include "bots/behavior_tree/priority/priority_node.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/action_node.h"
#include <memory>

#ifndef CSKNOW_TREE_H
#define CSKNOW_TREE_H


class Tree {
    // one order node overall, sets all team behavior
    std::unique_ptr<Blackboard> blackboard;
    std::unique_ptr<OrderSeqSelectorNode> orderNode;
    std::unique_ptr<PriorityNode> priorityNode;
    std::unique_ptr<ActionParSelectorNode> actionNode;
    set<CSGOId> lastFramePlayers;
    int32_t curMapNumber = INVALID_ID;

public:
    string curLog;
    void tick(ServerState & state, string mapsPath);
};

#endif //CSKNOW_TREE_H
