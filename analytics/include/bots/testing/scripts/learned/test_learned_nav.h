//
// Created by durst on 5/14/23.
//

#ifndef CSKNOW_TEST_LEARNED_NAV_H
#define CSKNOW_TEST_LEARNED_NAV_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"
#include "bots/analysis/learned_models.h"

namespace csknow::tests::learned {
    class LearnedNavScript : public Script {
    public:
        OrderId addedOrderId;
        size_t testIndex, numTests;
        bool waitForever;
        explicit LearnedNavScript(const std::string & name, size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state, Vec3 startPos, Vec2 startViewAngle,
                        const std::string & forceOrderNodeName, const vector<Waypoint> & waypoints,
                        const std::string & destinationPlace);
    };

    class LearnedGooseToCatScript : public LearnedNavScript {
    public:
        explicit LearnedGooseToCatScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class LearnedCTPushLongScript : public LearnedNavScript {
    public:
        explicit LearnedCTPushLongScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class LearnedCTPushBDoorsScript : public LearnedNavScript {
    public:
        explicit LearnedCTPushBDoorsScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class LearnedCTPushBHoleScript : public LearnedNavScript {
    public:
        explicit LearnedCTPushBHoleScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    vector<Script::Ptr> createLearnedNavScripts(size_t numTests, bool quitAtEnd);
}

#endif //CSKNOW_TEST_LEARNED_NAV_H
