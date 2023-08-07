//
// Created by durst on 5/9/22.
//

#ifndef CSKNOW_TREE_H
#define CSKNOW_TREE_H

#include "bots/behavior_tree/node.h"
#include "bots/behavior_tree/global/global_node.h"
#include "bots/behavior_tree/priority/priority_node.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/action/action_node.h"
#include "bots/analysis/inference_manager.h"
#include "bots/analysis/streaming_manager.h"
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#define ALL_PUSH false

class Tree {
public:
    // one order node overall, sets all team behavior
    std::unique_ptr<Blackboard> blackboard;
    std::unique_ptr<GlobalNode> globalNode;
    std::unique_ptr<PriorityNode> priorityNode;
    std::unique_ptr<ActionNode> actionNode;
    set<CSGOId> lastFramePlayers;
    int32_t curMapNumber = INVALID_ID, curRoundNumber = INVALID_ID;
    bool newBlackboard = false;
    bool resetState = false;
    TreeThinker defaultThinker{INVALID_ID, AggressiveType::Push};

    std::mutex filterMutex;
    set<string> sharedLogFilterNames, localLogFilterNames;

    string curLog;

    csknow::feature_store::FeatureStoreResult featureStoreResult;
    csknow::inference_manager::InferenceManager inferenceManager;

    Tree() { };
    Tree(const std::string & modelsDir) : inferenceManager(modelsDir) { };

    void tick(ServerState & state, const string & mapsPath);
    void readFilterNames();
};
void addTreeThinkersToBlackboard(const ServerState & state, Blackboard * blackboard);

#endif //CSKNOW_TREE_H
