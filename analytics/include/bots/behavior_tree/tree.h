//
// Created by durst on 5/9/22.
//

#ifndef CSKNOW_TREE_H
#define CSKNOW_TREE_H

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/global/global_node.h"
#include "bots/behavior_tree/priority/priority_node.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/action_node.h"
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>



class Tree {
public:
    // one order node overall, sets all team behavior
    std::unique_ptr<Blackboard> blackboard;
    std::unique_ptr<GlobalNode> globalNode;
    std::unique_ptr<PriorityNode> priorityNode;
    std::unique_ptr<ActionNode> actionNode;
    set<CSGOId> lastFramePlayers;
    int32_t curMapNumber = INVALID_ID;
    bool newBlackboard = false;
    bool resetState = false;
    TreeThinker defaultThinker{INVALID_ID};

    std::mutex filterMutex;
    set<string> sharedLogFilterNames, localLogFilterNames;

    string curLog;
    void tick(ServerState & state, const string & mapsPath);
    void readFilterNames();
};

#endif //CSKNOW_TREE_H
