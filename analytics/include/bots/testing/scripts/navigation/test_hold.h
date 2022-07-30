//
// Created by steam on 6/20/22.
//

#ifndef CSKNOW_HOLD_H
#define CSKNOW_HOLD_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"
#include "bots/testing/state_checks.h"

class HoldLongScript : public Script {
public:
    OrderId addedOrderId;

    HoldLongScript(const ServerState & state) :
        Script("HoldLongScript", {{0, ENGINE_TEAM_T}}, {ObserveType::FirstPerson, 0}) { }

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            Node::Ptr aimAtChoke = make_unique<RepeatDecorator>(blackboard,
                    make_unique<SelectorNode>(blackboard, Node::makeList(
                                              make_unique<PosConstraint>(blackboard, neededBots[0].id,
                                                                         PosConstraintDimension::Y, PosConstraintOp::GT,
                                                                         1000.),
                                              make_unique<AimingAtArea>(blackboard, vector{neededBots[0].id}, 3653))),
                                              false);
            Node::Ptr stillAndLookingAtChoke = make_unique<RepeatDecorator>(blackboard,
                                                                            make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                                    make_unique<StandingStill>(blackboard, vector{neededBots[0].id}),
                                                                                    make_unique<AimingAtArea>(blackboard, vector{neededBots[0].id}, 3653))),
                                                                true);
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1071.936035, 2972.308837, 128.762023}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ForceOrderNode>(blackboard, "ForceLongDefense", vector{neededBots[0].id}, strategy::defenseLongToAWaypoints, addedOrderId),
                                                         make_unique<ForceHoldIndexNode>(blackboard, "ForceAggroLong", vector{neededBots[0].id}, vector{4}, addedOrderId),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                 std::move(aimAtChoke),
                                                                 std::move(stillAndLookingAtChoke),
                                                                 make_unique<movement::WaitNode>(blackboard, 14, false)),
                                                            "HoldLongCondition")),
                                                 "HoldLongSequence");
        }
    }
};

class HoldASiteScript : public Script {
public:
    OrderId addedOrderId;

    HoldASiteScript(const ServerState & state) :
            Script("HoldASiteScript", {{0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}},
                   {ObserveType::Absolute, 0, {395.317963, 2659.722656, 559.311157}, {43.801949, -49.044704}}) { }

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);

            Node::Ptr stillAndLookingAtChoke = make_unique<RepeatDecorator>(blackboard,
                                                                            make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                                    make_unique<StandingStill>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}),
                                                                                    make_unique<AimingAtArea>(blackboard, vector{neededBots[0].id}, 3653),
                                                                                    make_unique<AimingAtArea>(blackboard, vector{neededBots[1].id}, 1384),
                                                                                    make_unique<AimingAtArea>(blackboard, vector{neededBots[2].id}, 4051))),
                                                                            true);

            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1241., 2586., 127.}), Vec2({0., 0.})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<TeleportPlantedC4>(blackboard),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1071.936035, 2972.308837, 128.762023}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard,neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({824.582764, 2612.630127, 95.957748}), Vec2({-1.760050, -105.049713})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard,neededBots[1].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1461.081055, 2392.754639, 22.165134}), Vec2({-89.683349, 0.746031})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[2].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<RecomputeOrdersNode>(blackboard),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                std::move(stillAndLookingAtChoke),
                                                                                                make_unique<movement::WaitNode>(blackboard, 14, false)),
                                                                                        "HoldASiteCondition")),
                                                 "HoldASiteSequence");
        }
    }
};
#endif //CSKNOW_HOLD_H
