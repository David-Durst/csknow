//
// Created by durst on 5/19/23.
//

#ifndef CSKNOW_TEST_TEAMWORK_H
#define CSKNOW_TEST_TEAMWORK_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"
#include "bots/analysis/learned_models.h"

namespace csknow::tests::learned {
    class LearnedTeamworkScript : public Script {
    public:
        OrderId addedOrderId;
        size_t testIndex, numTests;
        bool waitForever;

        explicit LearnedTeamworkScript(const std::string &name, vector<NeededBot> neededBots, ObserveSettings observeSettings,
                                   size_t testIndex, size_t numTests, bool waitForever);

        void initialize(Tree &tree, ServerState &state, vector<Vec3> playerPos, vector<Vec2> playerViewAngle, Vec3 c4Pos,
                        Node::Ptr forceSetup, Node::Ptr condition);
    };

    class LearnedPushLurkBaitASiteScript : public LearnedTeamworkScript {
        OrderId pushAddedOrderId, lurkAddedOrderId;
    public:
        explicit LearnedPushLurkBaitASiteScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class LearnedPushATwoOrdersScript : public LearnedTeamworkScript {
    public:
        explicit LearnedPushATwoOrdersScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class LearnedPushThreeBScript : public LearnedTeamworkScript {
    public:
        explicit LearnedPushThreeBScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    vector<Script::Ptr> createLearnedTeamworkScripts(size_t numTests, bool quitAtEnd);
}

#endif //CSKNOW_TEST_TEAMWORK_H
