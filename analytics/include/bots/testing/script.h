//
// Created by steam on 6/19/22.
//

#ifndef CSKNOW_SCRIPT_H
#define CSKNOW_SCRIPT_H

#include "bots/behavior_tree/node.h"
#include "bots/testing/command.h"
#include <memory>
using std::unique_ptr;

enum class ObserveType {
    FirstPerson,
    ThirdPerson,
    Absolute,
    None,
    NUM_OBSERVE_TYPE
};

struct ObserveValues {
    ObserveType observeType;
    CSGOId targetId;
    Vec3 cameraOrigin;
    Vec2 cameraPos;
};

struct NeededBot {
    CSGOId id;
    int team;
};

class Script {
    // children can't update blackboard
    unique_ptr<Blackboard> blackboard;
    vector<Command::Ptr> commands;
    TreeThinker treeThinker;
protected:
    vector<NeededBot> neededBots;
    vector<Command::Ptr> logicCommands;
    unique_ptr<Node> conditions;
    ObserveValues observeValues;

public:
    bool initialize(ServerState & state, string navPath);
    // prints result, returns when done
    bool tick(ServerState & state, string navPath);
};

#endif //CSKNOW_SCRIPT_H
