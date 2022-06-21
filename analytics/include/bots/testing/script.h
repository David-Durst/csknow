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

struct ObserveSettings {
    ObserveType observeType;
    CSKnowId neededBotIndex;
    Vec3 cameraOrigin;
    Vec2 cameraPos;
};

struct NeededBot {
    CSGOId id;
    int team;
};

class Script {
    // children can't update blackboard
    vector<Command::Ptr> commands;

protected:
    Blackboard & blackboard;
    string name;
    vector<NeededBot> neededBots;
    ObserveSettings observeSettings;
    vector<Command::Ptr> logicCommands;
    unique_ptr<Node> conditions;

public:
    Script(Blackboard & blackboard, string name, vector<NeededBot> neededBots, ObserveSettings observeSettings) :
            blackboard(blackboard), name(name), neededBots(neededBots), observeSettings(observeSettings) { }
           //vector<Command::Ptr> && logicCommands, unique_ptr<Node> && conditions) :
           //logicCommands(std::move(logicCommands)), conditions(std::move(conditions)) { }

    void initialize(ServerState & state, string navPath);
    virtual vector<string> generateCommands(ServerState & state);
    // prints result, returns when done
    bool tick(ServerState & state);
};

#endif //CSKNOW_SCRIPT_H
