//
// Created by durst on 5/14/23.
//

#ifndef CSKNOW_TEST_NAV_H
#define CSKNOW_TEST_NAV_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"
#include "bots/analysis/learned_models.h"

namespace csknow::tests::learned {
    class LearnedGooseToCatScript : public Script {
    public:
        OrderId addedOrderId;
        size_t testIndex, numTests;
        bool waitForever;
        explicit LearnedGooseToCatScript(size_t testIndex, size_t numTests, bool waitForever);
        void initialize(Tree & tree, ServerState & state) override;
    };

    vector<Script::Ptr> createLearnedNavScripts(size_t numTests, bool quitAtEnd);
}

#endif //CSKNOW_TEST_NAV_H
