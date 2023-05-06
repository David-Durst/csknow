//
// Created by durst on 5/6/23.
//

#ifndef CSKNOW_TEST_SETUP_H
#define CSKNOW_TEST_SETUP_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/tree.h"

class SetupCommands {
public:
    string botStopStr;
    int maxRounds;
    Node::Ptr commands;
    TreeThinker defaultThinker{INVALID_ID, AggressiveType::Push, {0., 0., 0., 0.}, 0.};

    explicit SetupCommands(string botStopStr, int maxRounds) :
        botStopStr(botStopStr), maxRounds(maxRounds), commands(nullptr) { }

    bool tick(ServerState & state, Blackboard & blackboard);
};

#endif //CSKNOW_TEST_SETUP_H
