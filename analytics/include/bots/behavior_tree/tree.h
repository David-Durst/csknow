//
// Created by durst on 5/9/22.
//

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/order_node.h"
#include "bots/behavior_tree/priority/priority_par_node.h"
#include "bots/behavior_tree/implementation_node.h"
#include "bots/behavior_tree/action_node.h"
#include <memory>

#ifndef CSKNOW_TREE_H
#define CSKNOW_TREE_H


class Tree {
    // one order node overall, sets all team behavior
    std::unique_ptr<Blackboard> blackboard;
    std::unique_ptr<OrderSeqSelectorNode> orderNode;
    vector<Node> perPlayerRootNodes;
    map<CSGOId, TreeThinker> playerToTreeThinkers;
    int32_t curMapNumber = INVALID_ID;

public:
    string curLog;
    void tick(ServerState & state, string mapsPath);
};

#endif //CSKNOW_TREE_H
