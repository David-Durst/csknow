//
// Created by steam on 8/3/22.
//

#ifndef CSKNOW_TEST_ENGAGE_SPACING_H
#define CSKNOW_TEST_ENGAGE_SPACING_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"

class CTEngageSpacingScript : public Script {
public:
    OrderId addedOrderId;

    CTEngageSpacingScript(const ServerState & state) :
            Script("PushLurkBaitASiteScript", {{0, ENGINE_TEAM_CT, AggressiveType::Push}, {0, ENGINE_TEAM_CT, AggressiveType::Push}, {0, ENGINE_TEAM_T}},
                   {ObserveType::FirstPerson, 1}) { };

    virtual void initialize(Tree & tree, ServerState & state) override  {
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
                                                                        make_unique<SetPos>(blackboard, Vec3({473.290436, -67.194908, 59.092133}), Vec2({-78.701942, -7.463999})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({-14.934761, -817.601318, 62.097897}), Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({1417.528564, 1652.856445, -9.775691}), Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[2].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                                        make_unique<ForceOrderNode>(blackboard, "ForceCTLongCat", vector{neededBots[0].id, neededBots[1].id}, strategy::offenseLongToAWaypoints, addedOrderId),
                                                                        make_unique<ForceEntryIndexNode>(blackboard, "ForcePusher",
                                                                                                         vector{neededBots[0].id, neededBots[1].id},
                                                                                                         vector{0, 1}),
                                                                        make_unique<movement::WaitNode>(blackboard, 2.0)),
                                                                "Setup");
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup",
                                                    vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}, false)
            ), "DisableDuringSetup");
            Node::Ptr attackersDifferentPlaces = make_unique<ParallelAndNode>(blackboard, Node::makeList(
                    make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[0].id, "LongA"), true),
                    make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[1].id, "LongDoors"), true)
            ));
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableEnemy", vector{neededBots[2].id}),
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisablePush", vector{neededBots[0].id, neededBots[1].id}, false, true, false),
                                                                                                // if the inner node doesn't finish in 15 seconds, fail right after
                                                                                                std::move(attackersDifferentPlaces),
                                                                                                make_unique<movement::WaitNode>(blackboard, 30, false)),
                                                                                        "CTEngageSpacingCondition")),
                                                 "CTEngageSpacingSequence");
        }
    }
};
#endif //CSKNOW_TEST_ENGAGE_SPACING_H
