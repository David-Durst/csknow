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
                                                                 make_unique<movement::WaitNode>(blackboard, 20, false)),
                                                            "HoldLongCondition")),
                                                 "HoldLongSequence");
        }
    }
};

class HoldASitePushScript : public Script {
public:
    HoldASitePushScript(const ServerState & state) :
            Script("HoldASitePushScript", {{0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}},
                   {ObserveType::Absolute, 0, {395.317963, 2659.722656, 559.311157}, {43.801949, -49.044704}}) { }

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);

            auto neededBotIds = vector{neededBots[0].id, neededBots[1].id, neededBots[2].id};
            Node::Ptr stillAndLookingAtChoke = make_unique<RepeatDecorator>(blackboard,
                                                                            make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                                    make_unique<StandingStill>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}),
                                                                                    make_unique<AimingAtArea>(blackboard, neededBotIds, 3653),
                                                                                    make_unique<AimingAtArea>(blackboard, neededBotIds, 1384),
                                                                                    make_unique<AimingAtArea>(blackboard, neededBotIds, 4051))),
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
                                                                                        "HoldASitePushCondition")),
                                                 "HoldASitePushSequence");
        }
    }
};

class HoldASiteBaitScript : public Script {
public:
    HoldASiteBaitScript(const ServerState & state) :
            Script("HoldASiteBaitScript", {{0, ENGINE_TEAM_T, AggressiveType::Bait}, {0, ENGINE_TEAM_T, AggressiveType::Bait}, {0, ENGINE_TEAM_T, AggressiveType::Bait}},
                   {ObserveType::Absolute, 0, {395.317963, 2659.722656, 559.311157}, {43.801949, -49.044704}}) { }

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);

            auto neededBotIds = vector{neededBots[0].id, neededBots[1].id, neededBots[2].id};
            Node::Ptr stillAndLookingAtChoke = make_unique<RepeatDecorator>(blackboard,
                                                                            make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                                    make_unique<StandingStill>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}),
                                                                                    make_unique<AimingAtArea>(blackboard, neededBotIds, 4170),
                                                                                    make_unique<AimingAtArea>(blackboard, neededBotIds, 9018),
                                                                                    make_unique<AimingAtArea>(blackboard, neededBotIds, 4048))),
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
                                                                                        "HoldASiteBaitCondition")),
                                                 "HoldASiteBaitSequence");
        }
    }
};

class HoldBSitePushScript : public Script {
public:
    HoldBSitePushScript(const ServerState & state) :
            Script("HoldBSitePushScript", {{0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_T}, {0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
                   {ObserveType::Absolute, 0, {-2092., 3050., 710.}, {56., -68.}}) { }

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);

            auto neededBotIds = vector{neededBots[0].id, neededBots[1].id, neededBots[2].id};
            Node::Ptr stillAndLookingAtChoke = make_unique<RepeatDecorator>(blackboard,
                                                                            make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                                    make_unique<StandingStill>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}),
                                                                                    make_unique<AimingAtArea>(blackboard, neededBotIds, 1896),
                                                                                    make_unique<AimingAtArea>(blackboard, neededBotIds, 8489),
                                                                                    make_unique<AimingAtArea>(blackboard, neededBotIds, 556))),
                                                                            true);

            vector<AreaId> emptyVector{}, onePlayerRequiredNotPossibleAreas{8083};
            Node::Ptr validConditions = make_unique<ParallelAndNode>(blackboard, Node::makeList(
                    std::move(stillAndLookingAtChoke),
                    make_unique<CheckPossibleLocationsNode>(blackboard, vector{neededBots[3].id, neededBots[4].id},
                                                            vector{emptyVector, emptyVector}, // just checking that nothing slips by bot, see https://www.youtube.com/watch?v=wJEu_NLne40 for example
                                                            vector{onePlayerRequiredNotPossibleAreas, onePlayerRequiredNotPossibleAreas})
                ));

            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id, neededBots[3].id, neededBots[4].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({-1574., 2638., 38.}), Vec2({0., 0.})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<TeleportPlantedC4>(blackboard),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({-1990., 2644., 93.6}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard,neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({-1438., 2461., 65.5}), Vec2({-1.760050, -105.049713})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard,neededBots[1].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({-1750., 1868., 64.2}), Vec2({-89.683349, 0.746031})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[2].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({-516., 1733., -14.}), Vec2({-89.683349, 0.746031})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[3].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({-1591., 200., 129.}), Vec2({-89.683349, 0.746031})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[4].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<RecomputeOrdersNode>(blackboard),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<DisableActionsNode>(blackboard, "disableAll", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id, neededBots[3].id, neededBots[4].id}),
                                                                                                make_unique<movement::WaitNode>(blackboard, 2.0))),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<DisableActionsNode>(blackboard, "disableOffense", vector{neededBots[3].id, neededBots[4].id}),
                                                                                                make_unique<SavePossibleVisibleOverlays>(blackboard, vector{neededBots[3].id, neededBots[4].id}, false),
                                                                                                std::move(validConditions),
                                                                                                make_unique<movement::WaitNode>(blackboard, 14, false)),
                                                                                        "HoldBSitePushCondition")),
                                                 "HoldBSitePushSequence");
        }
    }
};

class HoldBSiteBaitScript : public Script {
public:
    HoldBSiteBaitScript(const ServerState & state) :
            Script("HoldBSiteBaitScript", {{0, ENGINE_TEAM_T, AggressiveType::Bait}, {0, ENGINE_TEAM_T, AggressiveType::Bait}, {0, ENGINE_TEAM_T, AggressiveType::Bait}},
                   {ObserveType::Absolute, 0, {-2092., 3050., 710.}, {56., -68.}}) { }

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);

            auto neededBotIds = vector{neededBots[0].id, neededBots[1].id, neededBots[2].id};
            Node::Ptr stillAndLookingAtChoke = make_unique<RepeatDecorator>(blackboard,
                                                                            make_unique<SequenceNode>(blackboard, Node::makeList(
                                                                                    make_unique<StandingStill>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}),
                                                                                    make_unique<AimingAtArea>(blackboard, neededBotIds, 7533),
                                                                                    make_unique<AimingAtArea>(blackboard, neededBotIds, 8666),
                                                                                    make_unique<AimingAtArea>(blackboard, neededBotIds, 8212))),
                                                                            true);

            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({-1574., 2638., 38.}), Vec2({0., 0.})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<TeleportPlantedC4>(blackboard),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({-1990., 2644., 93.6}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard,neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({-1438., 2461., 65.5}), Vec2({-1.760050, -105.049713})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard,neededBots[1].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({-1750., 1868., 64.2}), Vec2({-89.683349, 0.746031})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[2].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<RecomputeOrdersNode>(blackboard),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                std::move(stillAndLookingAtChoke),
                                                                                                make_unique<movement::WaitNode>(blackboard, 14, false)),
                                                                                        "HoldBSiteBaitCondition")),
                                                 "HoldBSiteBaitSequence");
        }
    }
};
#endif //CSKNOW_HOLD_H
