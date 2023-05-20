//
// Created by durst on 5/18/23.
//

#ifndef CSKNOW_TEST_LEARNED_HOLD_H
#define CSKNOW_TEST_LEARNED_HOLD_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"
#include "bots/analysis/learned_models.h"

namespace csknow::tests::learned {
    class LearnedHoldScript : public Script {
    public:
        OrderId addedOrderId;
        size_t testIndex, numTests;
        bool waitForever;

        explicit LearnedHoldScript(const std::string &name, vector<NeededBot> neededBots, ObserveSettings observeSettings,
                                   size_t testIndex, size_t numTests, bool waitForever);

        void initialize(Tree &tree, ServerState &state, vector<Vec3> playerPos,
                        vector<Vec2> playerViewAngle, Vec3 c4Pos, vector<AreaId> chokeAreas);
    };

    class LearnedHoldASitePushScript : public LearnedHoldScript {
    public:
        explicit LearnedHoldASitePushScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class LearnedHoldASiteBaitScript : public LearnedHoldScript {
    public:
        explicit LearnedHoldASiteBaitScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class LearnedHoldBSitePushScript : public LearnedHoldScript {
    public:
        explicit LearnedHoldBSitePushScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    class LearnedHoldBSiteBaitScript : public LearnedHoldScript {
    public:
        explicit LearnedHoldBSiteBaitScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    vector<Script::Ptr> createLearnedHoldScripts(size_t numTests, bool quitAtEnd);
}

#endif //CSKNOW_TEST_LEARNED_HOLD_H
