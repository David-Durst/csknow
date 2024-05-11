//
// Created by durst on 5/10/24.
//

#ifndef CSKNOW_TEST_ORAL_EXAMPLES_H
#define CSKNOW_TEST_ORAL_EXAMPLES_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"

class PositionCoverScript : public Script {
public:
    OrderId addedOrderId;

    explicit PositionCoverScript(const ServerState &) :
            Script("PositionCoverScript", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_T}},
                   {ObserveType::Absolute, 0, {1140.026000, 2167.126708, 233.134750}, {38.555763, 162.810623}}) { }

    void initialize(Tree & tree, ServerState & state) override {
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
                                                                        make_unique<SetPos>(blackboard, Vec3({949.075134, 2370.172363, 5.394318}), Vec2({90, -57.375949})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({925.600219, 2541.894775, 95.896392}), Vec2({-90.638359, 14.121987})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                                        make_unique<movement::WaitNode>(blackboard, 40.0)),
                                                                "DangerSetup");
            /*
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id, neededBots[3].id}, false)
            ), "DangerDisableDuringSetup");
             */

            /*
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id}),
                                                                                                // if the inner node doesn't finish in 15 seconds, fail right after
                                                                                                make_unique<movement::WaitNode>(blackboard, 36, false)),
                                                                                        "DangerTwoPlayerCondition")),
                                                 "PositionCoverSequence");
                                                 */
            commands = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                         make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id}),
                                                         std::move(setupCommands)),
                                                 "PositionCoverSequence");
        }
    }
};

class PositionVisibilityScript : public Script {
public:
    OrderId addedOrderId;

    explicit PositionVisibilityScript(const ServerState &) :
            Script("PositionVisibilityScript", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_T}},
                   {ObserveType::Absolute, 0, {1140.026000, 2167.126708, 233.134750}, {38.555763, 162.810623}}) { }

    void initialize(Tree & tree, ServerState & state) override {
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
                                                                        make_unique<SetPos>(blackboard, Vec3({930.029357, 2039.893920, -13.652070}), Vec2({90.688308, -12.040013})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({925.600219, 2541.894775, 95.896392}), Vec2({-90.638359, 14.121987})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                                        make_unique<movement::WaitNode>(blackboard, 40.0)),
                                                                "DangerSetup");
            /*
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id, neededBots[3].id}, false)
            ), "DangerDisableDuringSetup");
             */

            /*
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id}),
                                                                                                // if the inner node doesn't finish in 15 seconds, fail right after
                                                                                                make_unique<movement::WaitNode>(blackboard, 36, false)),
                                                                                        "DangerTwoPlayerCondition")),
                                                 "PositionVisibilitySequence");
                                                 */
            commands = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                              make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id}),
                                                              std::move(setupCommands)),
                                                      "PositionVisibilitySequence");
        }
    }
};

#endif //CSKNOW_TEST_ORAL_EXAMPLES_H
