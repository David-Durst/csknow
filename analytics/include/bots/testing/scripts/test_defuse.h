//
// Created by steam on 8/4/22.
//

#ifndef CSKNOW_TEST_DEFUSE_H
#define CSKNOW_TEST_DEFUSE_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"

class DefuseScript : public Script {
public:
    OrderId addedOrderId;

    DefuseScript(const ServerState & state) :
            Script("DefuseScript", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
                   {ObserveType::Absolute, 0, {395.317963, 2659.722656, 559.311157}, {43.801949, -49.044704}}) { }

    virtual void initialize(Tree & tree, ServerState & state) override {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                        make_unique<InitTestingRound>(blackboard, name),
                                                                        make_unique<movement::WaitNode>(blackboard, 1.0),
                                                                        make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id}, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.3),
                                                                        make_unique<SetPos>(blackboard, Vec3({1241., 2586., 127.}), Vec2({0., 0.})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<TeleportPlantedC4>(blackboard),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({367.184967, 1543.949097, 2.625248}), Vec2({-78.701942, -7.463999})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({246.444977, 1385.060669, 0.139538}), Vec2({90., 0.})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                                        make_unique<ForceOrderNode>(blackboard, "ForceCTCat", vector{neededBots[0].id, neededBots[1].id}, strategy::offenseCatToAWaypoints, addedOrderId),
                                                                        make_unique<ForceEntryIndexNode>(blackboard, "ForcePusherBaiter",
                                                                                                         vector{neededBots[0].id, neededBots[1].id},
                                                                                                         vector{0, 1}),
                                                                        make_unique<ForceDefuser>(blackboard, neededBots[0].id),
                                                                        make_unique<movement::WaitNode>(blackboard, 2.0)),
                                                                "DefuserSetup");
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id}, false)
            ), "DefuserDisableDuringSetup");

            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                 make_unique<RepeatDecorator>(blackboard, make_unique<C4Defused>(blackboard), true),
                                                                 make_unique<movement::WaitNode>(blackboard, 26, false)),
                                                            "DefuserCondition")),
                                                 "DefuserSequence");
        }
    }
};

#endif //CSKNOW_TEST_DEFUSE_H
