//
// Created by steam on 7/12/22.
//

#ifndef CSKNOW_TEST_DANGER_H
#define CSKNOW_TEST_DANGER_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"

class DangerCheck : public Script {
public:
    DangerCheck(const ServerState & state) :
            Script("DangerCheck", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}},
                   {ObserveType::Absolute, 0, {395.317963, 2659.722656, 559.311157}, {43.801949, -49.044704}}) { }

    virtual void initialize(Tree & tree, ServerState & state) override {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            vector<string> aToCatPathPlace(order::catToAPathPlace.rbegin(), order::catToAPathPlace.rend());
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                        make_unique<InitTestingRound>(blackboard, name),
                                                                        make_unique<movement::WaitNode>(blackboard, 1.0),
                                                                        make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id, neededBots[3].id}, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.3),
                                                                        make_unique<SetPos>(blackboard, Vec3({367.184967, 1543.949097, 2.625248}), Vec2({-78.701942, -7.463999})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({246.444977, 1385.060669, 0.139538}), Vec2({90., 0.})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -9.775691}), Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[2].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1461.081055, 2392.754639, 22.165134}), Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[3].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                                        make_unique<ForceOrderNode>(blackboard, "ForceCTCat", vector{neededBots[0].id, neededBots[1].id}, order::catToAPathPlace),
                                                                        make_unique<movement::WaitNode>(blackboard, 2.0)),
                                                                "DangerSetup");
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id, neededBots[3].id}, false)
            ), "DangerDisableDuringSetup");
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                //make_unique<AimingAt>(blackboard, neededBots[0].id, neededBots[2].id),
                                                                                                //make_unique<Firing>(blackboard, neededBots[0].id, true),
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[2].id, neededBots[3].id}),
                                                                                                make_unique<movement::WaitNode>(blackboard, 20)),
                                                                 //make_unique<movement::WaitNode>(blackboard, 0.8)),
                                                                                        "DangerCondition")),
                                                 "DangerSequence");
        }
    }
};

#endif //CSKNOW_TEST_DANGER_H
