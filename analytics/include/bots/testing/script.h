//
// Created by steam on 6/19/22.
//

#ifndef CSKNOW_SCRIPT_H
#define CSKNOW_SCRIPT_H

#include "bots/behavior_tree/node.h"
#include "bots/testing/command.h"
#include <memory>

enum class ObserveType {
    FirstPerson,
    ThirdPerson,
    Absolute,
    None,
    NUM_OBSERVE_TYPE
};

struct ObserveValues {
    ObserveType observeType;
    CSKnowId targetId;
    Vec3 cameraOrigin;
    Vec2 cameraPos;
};

class Script {
    // one order node overall, sets all team behavior
    std::unique_ptr<Blackboard> blackboard;
    std::vector<Command::Ptr> commands;
    std::unique_ptr<Node> conditions;
    ObserveValues observeValues;

public:
    Script(std::unique_ptr<Blackboard> && blackboard, std::vector<Command::Ptr> && commands, std::unique_ptr<Node> && conditions, ObserveValues observeValues) :
        blackboard(std::move(blackboard)), commands(std::move(commands)), conditions(std::move(conditions)), observeValues(observeValues) { }
    void tick(ServerState & state);
};

#endif //CSKNOW_SCRIPT_H
