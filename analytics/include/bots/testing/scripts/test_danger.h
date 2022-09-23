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

class DangerOnePlayerCheck : public Script {
public:
    OrderId addedOrderId;

    explicit DangerOnePlayerCheck(const ServerState &) :
            Script("DangerOnePlayerCheck", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}},
                   {ObserveType::Absolute, 0, {395.317963, 2659.722656, 559.311157}, {43.801949, -49.044704}}) { }

    void initialize(Tree & tree, ServerState & state) override {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                        make_unique<InitTestingRound>(blackboard, name),
                                                                        make_unique<movement::WaitNode>(blackboard, 1.0),
                                                                        make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.3),
                                                                        make_unique<SetPos>(blackboard, Vec3({367.184967, 1543.949097, 2.625248}), Vec2({-78.701942, -7.463999})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -9.775691}), Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1461.081055, 2392.754639, 22.165134}), Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[2].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                                        make_unique<ForceOrderNode>(blackboard, "ForceCTCat", vector{neededBots[0].id}, testCatToAWaypoints, addedOrderId),
                                                                        make_unique<movement::WaitNode>(blackboard, 2.0)),
                                                                "DangerSetup");
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}, false)
            ), "DangerDisableDuringSetup");

            Node::Ptr areasToCheck = make_unique<ParallelAndNode>(blackboard, Node::makeList(
                    make_unique<RepeatDecorator>(blackboard, make_unique<AimingAtArea>(blackboard, vector{neededBots[0].id}, 4201), true),
                    make_unique<RepeatDecorator>(blackboard, make_unique<AimingAtArea>(blackboard, vector{neededBots[0].id}, 8672), true),
                    make_unique<RepeatDecorator>(blackboard, make_unique<AimingAtArea>(blackboard, vector{neededBots[0].id}, 1399), true)
            ), "areasToCheck");
            Node::Ptr lastLongEnoughForDifferentDangerNodes = make_unique<ParallelAndNode>(blackboard, Node::makeList(
                    std::move(areasToCheck),
                    // this is just to keep this node running, outer time is for failing if don't complete
                    make_unique<movement::WaitNode>(blackboard, 15)
            ), "lastForDifferentDangerAreasCheck");
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                std::move(lastLongEnoughForDifferentDangerNodes),
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[1].id, neededBots[2].id}),
                                                                                                // if the inner node doesn't finish in 15 seconds, fail right after
                                                                                                make_unique<movement::WaitNode>(blackboard, 16, false)),
                                                                                        "DangerOnePlayerCondition")),
                                                 "DangerOnePlayerSequence");
        }
    }
};

class DangerTwoPlayerCheck : public Script {
public:
    OrderId addedOrderId;

    explicit DangerTwoPlayerCheck(const ServerState &) :
            Script("DangerTwoPlayerCheck", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}},
                   {ObserveType::Absolute, 0, {395.317963, 2659.722656, 559.311157}, {43.801949, -49.044704}}) { }

    void initialize(Tree & tree, ServerState & state) override {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
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
                                                                        make_unique<ForceOrderNode>(blackboard, "ForceCTCat", vector{neededBots[0].id, neededBots[1].id}, testCatToAWaypoints, addedOrderId),
                                                                        make_unique<ForceEntryIndexNode>(blackboard, "ForcePusherBaiter",
                                                                                                         vector{neededBots[0].id, neededBots[1].id},
                                                                                                         vector{0, 1}),
                                                                        make_unique<movement::WaitNode>(blackboard, 2.0)),
                                                                "DangerSetup");
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id, neededBots[3].id}, false)
            ), "DangerDisableDuringSetup");

            Node::Ptr areasToCheck = make_unique<ParallelAndNode>(blackboard, Node::makeList(
                    make_unique<RepeatDecorator>(blackboard, make_unique<AimingAtArea>(blackboard, vector{neededBots[0].id, neededBots[1].id}, 4201), true),
                    make_unique<RepeatDecorator>(blackboard, make_unique<AimingAtArea>(blackboard, vector{neededBots[0].id, neededBots[1].id}, 8672), true),
                    make_unique<RepeatDecorator>(blackboard, make_unique<AimingAtArea>(blackboard, vector{neededBots[0].id, neededBots[1].id}, 1399), true)
            ), "areasToCheck");
            // different areas check is wonky
            Node::Ptr lastLongEnoughForDifferentDangerNodes = make_unique<ParallelAndNode>(blackboard, Node::makeList(
                    std::move(areasToCheck),
                    /*
                    make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                            make_unique<RepeatDecorator>(blackboard, make_unique<RequireDifferentDangerAreasNode>(blackboard, vector{neededBots[0].id, neededBots[1].id}), false),
                            // this is just to keep this node running, outer time is for failing if don't complete
                            make_unique<movement::WaitNode>(blackboard, 15)
                    )),
                     */
                    make_unique<movement::WaitNode>(blackboard, 15)
            ), "lastForDifferentDangerAreasCheck");
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                std::move(lastLongEnoughForDifferentDangerNodes),
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[2].id, neededBots[3].id}),
                                                                                                // if the inner node doesn't finish in 15 seconds, fail right after
                                                                                                make_unique<movement::WaitNode>(blackboard, 16, false)),
                                                                                        "DangerTwoPlayerCondition")),
                                                 "DangerTwoPlayerSequence");
        }
    }
};

#endif //CSKNOW_TEST_DANGER_H
