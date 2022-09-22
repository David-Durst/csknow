//
// Created by steam on 6/30/22.
//

#ifndef CSKNOW_TEAMWORK_H
#define CSKNOW_TEAMWORK_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"
#include "bots/testing/blackboard_management.h"

class PusherReachesBeforeBaiter : public Node {
    CSGOId pusherId, baiterId;
    string pusherFinalPlace;
    set<string> baiterValidPlaces;

public:
    PusherReachesBeforeBaiter(Blackboard & blackboard, CSGOId pusherId, CSGOId baiterId, string pusherFinalPlace, set<string> baiterValidPlaces) :
            Node(blackboard, "ValidConditionNode"), pusherId(pusherId), baiterId(baiterId), pusherFinalPlace(pusherFinalPlace), baiterValidPlaces(baiterValidPlaces) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & pusherClient = state.getClient(pusherId);
        uint32_t pusherArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(pusherClient.getFootPosForPlayer())).get_id();
        string pusherPlace = blackboard.navFile.get_place(blackboard.navFile.get_area_by_id_fast(pusherArea).m_place);

        const ServerState::Client & baiterClient = state.getClient(baiterId);
        uint32_t baiterArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(baiterClient.getFootPosForPlayer())).get_id();
        string baiterPlace = blackboard.navFile.get_place(blackboard.navFile.get_area_by_id_fast(baiterArea).m_place);

        if (pusherPlace == pusherFinalPlace) {
            playerNodeState[treeThinker.csgoId] = (baiterValidPlaces.find(baiterPlace) != baiterValidPlaces.end()) ? NodeState::Success : NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
            return NodeState::Running;
        }
    }
};

class PushBaitGooseToCatScript : public Script {
public:
    OrderId addedOrderId;

    PushBaitGooseToCatScript(const ServerState & state) :
            Script("PushBaitGooseToCatScript", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
                   {ObserveType::Absolute, 0, {366.774475, 2669.538818, 239.860245}, {16.486465, -46.266056}}) { };

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<string> baiterValidLocations{"ShortStairs"};
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1071.936035, 2972.308837, 128.762023}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard,neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({824.582764, 2612.630127, 95.957748}), Vec2({-1.760050, -105.049713})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ForceOrderNode>(blackboard, "ForceTCat", vector{neededBots[0].id, neededBots[1].id}, testAToCatWaypoints, addedOrderId),
                                                         make_unique<ForceEntryIndexNode>(blackboard, "ForcePusherBaiter",
                                                                                          vector{neededBots[0].id, neededBots[1].id},
                                                                                          vector{0, 1}),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<PusherReachesBeforeBaiter>(blackboard, neededBots[0].id, neededBots[1].id, "Catwalk", baiterValidLocations),
                                                                                                make_unique<movement::WaitNode>(blackboard, 20, false)),
                                                                                        "PushBaitGooseToCatCondition")),
                                                 "PushBaitGooseToCatSequence");
        }
    }
};

class PushWaitForBaitGooseToCatScript : public Script {
public:
    OrderId addedOrderId;

    PushWaitForBaitGooseToCatScript(const ServerState & state) :
            Script("PushWaitForBaitGooseToCatScript", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
                   {ObserveType::Absolute, 0, {366.774475, 2669.538818, 239.860245}, {16.486465, -46.266056}}) { };

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<string> baiterValidLocations{"ShortStairs"};
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({420.199219, 2377.000000, 96.528168}), Vec2({-0.659997, 5.090078})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard,neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1071.936035, 2972.308837, 128.762023}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ForceOrderNode>(blackboard, "ForceTCat", vector{neededBots[0].id, neededBots[1].id}, testAToCatWaypoints, addedOrderId),
                                                         make_unique<ForceEntryIndexNode>(blackboard, "ForcePusherBaiter",
                                                                                          vector{neededBots[0].id, neededBots[1].id},
                                                                                          vector{0, 1}),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<PusherReachesBeforeBaiter>(blackboard, neededBots[0].id, neededBots[1].id, "Catwalk", baiterValidLocations),
                                                                                                make_unique<movement::WaitNode>(blackboard, 20, false))
                                                                                        ))
                                                 );
        }
    }
};

class PushMultipleBaitGooseToCatScript : public Script {
public:
    OrderId addedOrderId;

    PushMultipleBaitGooseToCatScript(const ServerState & state) :
            Script("PushMultipleBaitGooseToCatScript", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
                   {ObserveType::Absolute, 0, {366.774475, 2669.538818, 239.860245}, {16.486465, -46.266056}}) { };

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<string> baiter0ValidLocations{"ShortStairs"}, baiter1ValidLocations{"ExtendedA", ""};
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1071.936035, 2972.308837, 128.762023}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard,neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({824.582764, 2612.630127, 95.957748}), Vec2({-1.760050, -105.049713})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({420.199219, 2377.000000, 96.528168}), Vec2({-0.659997, 5.090078})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[2].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ForceOrderNode>(blackboard, "ForceTCat", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}, testAToCatWaypoints, addedOrderId),
                                                         make_unique<ForceEntryIndexNode>(blackboard, "ForcePusherBaiter",
                                                                                          vector{neededBots[0].id, neededBots[1].id, neededBots[2].id},
                                                                                          vector{0, 1, 2}),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<ParallelAndNode>(blackboard, Node::makeList(
                                                                                                        make_unique<PusherReachesBeforeBaiter>(blackboard, neededBots[0].id, neededBots[1].id, "Catwalk", baiter0ValidLocations),
                                                                                                        make_unique<PusherReachesBeforeBaiter>(blackboard, neededBots[1].id, neededBots[2].id, "ShortStairs", baiter1ValidLocations)
                                                                                                        ), "PushMulitpleBaitLocations"),
                                                                                                make_unique<movement::WaitNode>(blackboard, 20, false)),
                                                                                        "PushMultipleBaitGooseToLongCondition")),
                                                 "PushMultipleBaitGooseToLongSequence");
        }
    }
};

class PushLurkBaitASiteScript : public Script {
public:
    OrderId pushAddedOrderId, lurkAddedOrderId;

    PushLurkBaitASiteScript(const ServerState & state) :
            Script("PushLurkBaitASiteScript", {{0, ENGINE_TEAM_CT, AggressiveType::Bait}, {0, ENGINE_TEAM_CT, AggressiveType::Bait}, {0, ENGINE_TEAM_CT, AggressiveType::Push}, {0, ENGINE_TEAM_T}},
                   {ObserveType::FirstPerson, 0}) { };

    virtual void initialize(Tree & tree, ServerState & state) override  {
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
                                                                                //;setang -1.012000 67.385880
                                                                        make_unique<SetPos>(blackboard, Vec3({473.290436, -67.194908, 59.092133}), Vec2({-78.701942, -7.463999})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[0].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({-14.934761, -817.601318, 62.097897}), Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({-421.860260, 856.695313, 42.407509}), Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[2].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<SetPos>(blackboard, Vec3({883.084106, 2491.471436, 160.187653}), Vec2({-89.683349, 0.746031})),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<Teleport>(blackboard, neededBots[3].id, state),
                                                                        make_unique<movement::WaitNode>(blackboard, 0.1),
                                                                        make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                                        make_unique<ForceOrderNode>(blackboard, "ForceCTLongCat", vector{neededBots[0].id, neededBots[1].id}, strategy::offenseLongToAWaypoints(), lurkAddedOrderId),
                                                                        make_unique<ForceOrderNode>(blackboard, "ForceCTCat", vector{neededBots[2].id}, strategy::offenseCatToAWaypoints, pushAddedOrderId),
                                                                        make_unique<ForceEntryIndexNode>(blackboard, "ForcePusherBaiter",
                                                                                                         vector{neededBots[0].id, neededBots[1].id, neededBots[2].id},
                                                                                                         vector{0, 1, 0}),
                                                                        make_unique<movement::WaitNode>(blackboard, 2.0)),
                                                                "Setup");
            Node::Ptr disableAllBothDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup",
                                                    vector{neededBots[0].id, neededBots[1].id, neededBots[2].id, neededBots[3].id}, false)
            ), "DisableDuringSetup");
            Node::Ptr pusherSeesEnemyBeforeLurkerMoves = make_unique<ParallelAndNode>(blackboard, Node::makeList(
                    make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[0].id, "LongDoors"), true),
                    make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[1].id, "OutsideLong"), true),
                    make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[2].id, "ExtendedA"), true)
            ));
            Node::Ptr pusherHoldsLurkerReachesLongBaiterFollows = make_unique<ParallelAndNode>(blackboard, Node::makeList(
                    make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[0].id, "LongA"), true),
                    make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[1].id, "LongA"), true),
                    make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[2].id, "ExtendedA"), true)
            ));
            Node::Ptr placeChecks = make_unique<SequenceNode>(blackboard, Node::makeList(
                    std::move(pusherSeesEnemyBeforeLurkerMoves),
                    std::move(pusherHoldsLurkerReachesLongBaiterFollows)
            ));
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         std::move(disableAllBothDuringSetup),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisableEnemy", vector{neededBots[3].id}),
                                                                                                make_unique<DisableActionsNode>(blackboard, "DisablePush", vector{neededBots[2].id}, false, true, false),
                                                                                                // if the inner node doesn't finish in 15 seconds, fail right after
                                                                                                std::move(placeChecks),
                                                                                                make_unique<movement::WaitNode>(blackboard, 30, false)),
                                                                                        "PushLurkBaitCondition")),
                                                 "PushLurkBaitSequence");
        }
    }
};

class PushATwoOrdersScript : public Script {
public:
    OrderId addedOrderId;

    PushATwoOrdersScript(const ServerState & state) :
            Script("PushATwoOrdersScript", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
                   {ObserveType::FirstPerson, 1}) { };

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<string> baiterValidLocations{"ShortStairs", "ExtendedA"};
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({-376., 729., 64.}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard,neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({-944., 1440., -49.}), Vec2({-1.760050, -105.049713})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1241., 2586., 127.}), Vec2({0., 0.})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<TeleportPlantedC4>(blackboard),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                                                         make_unique<RecomputeOrdersNode>(blackboard),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<PusherReachesBeforeBaiter>(blackboard, neededBots[1].id, neededBots[0].id, "UnderA", baiterValidLocations),
                                                                                                make_unique<movement::WaitNode>(blackboard, 20, false))
                                                                                        ))
                                                 );
        }
    }
};

class PushTwoBDoorsScript : public Script {
public:
    PushTwoBDoorsScript(const ServerState & state) :
            Script("PushTwoBDoorsScript", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
                   {ObserveType::FirstPerson, 0}) { };

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<string> baiterValidLocations{"BDoors", "BombsiteB"};
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                    make_unique<InitTestingRound>(blackboard, name),
                    make_unique<movement::WaitNode>(blackboard, 1.0),
                    make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id},state),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    //make_unique<SetPos>(blackboard, Vec3({-157., 1380., 64.}), Vec2({2.903987, -95.587982})),
                    make_unique<SetPos>(blackboard, Vec3({-224., 1225., 66.}), Vec2({2.903987, -95.587982})),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<Teleport>(blackboard,neededBots[0].id, state),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<SetPos>(blackboard, Vec3({344., 2292., -118.}), Vec2({-1.760050, -105.049713})),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<Teleport>(blackboard, neededBots[1].id, state),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<SetPos>(blackboard, Vec3({-1463., 2489., 46.}), Vec2({0., 0.})),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<TeleportPlantedC4>(blackboard),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                    make_unique<RecomputeOrdersNode>(blackboard)));
            Node::Ptr disableAllDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id})
            ), "DisableDuringSetup");
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                    std::move(disableAllDuringSetup),
                    make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                            make_unique<RepeatDecorator>(blackboard, make_unique<DistanceConstraint>(blackboard, neededBots[0].id, neededBots[1].id, PosConstraintOp::LT, 300.), true),
                            make_unique<movement::WaitNode>(blackboard, 15, false))
                    ))
            );
        }
    }
};

class PushThreeBScript : public Script {
public:
    PushThreeBScript(const ServerState & state) :
            Script("PushThreeBScript", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
                   {ObserveType::FirstPerson, 0}) { };

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<string> baiterValidLocations{"BDoors", "BombsiteB"};
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                    make_unique<InitTestingRound>(blackboard, name),
                    make_unique<movement::WaitNode>(blackboard, 1.0),
                    make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id},state),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    //make_unique<SetPos>(blackboard, Vec3({-157., 1380., 64.}), Vec2({2.903987, -95.587982})),
                    make_unique<SetPos>(blackboard, Vec3({-224., 1225., 66.}), Vec2({2.903987, -95.587982})),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<Teleport>(blackboard,neededBots[0].id, state),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<SetPos>(blackboard, Vec3({344., 2292., -118.}), Vec2({-1.760050, -105.049713})),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<Teleport>(blackboard, neededBots[1].id, state),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<SetPos>(blackboard, Vec3({-1591., 200., 129.}), Vec2({-89.683349, 0.746031})),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<Teleport>(blackboard, neededBots[2].id, state),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<SetPos>(blackboard, Vec3({-1463., 2489., 46.}), Vec2({0., 0.})),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<TeleportPlantedC4>(blackboard),
                    make_unique<movement::WaitNode>(blackboard, 0.1),
                    make_unique<ClearMemoryCommunicationDangerNode>(blackboard),
                    make_unique<RecomputeOrdersNode>(blackboard)));
            Node::Ptr disableAllDuringSetup = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                    std::move(setupCommands),
                    make_unique<DisableActionsNode>(blackboard, "DisableSetup", vector{neededBots[0].id, neededBots[1].id})
            ), "DisableDuringSetup");
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                    std::move(disableAllDuringSetup),
                    make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                            make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[0].id, "BombsiteB"), true),
                            make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[1].id, "BombsiteB"), true),
                            make_unique<RepeatDecorator>(blackboard, make_unique<InPlace>(blackboard, neededBots[2].id, "BombsiteB"), true),
                            make_unique<movement::WaitNode>(blackboard, 35, false))
                    ))
            );
        }
    }
};
/*
class TmpPushMultipleBaitGooseToCatScript : public Script {
public:
    OrderId addedOrderId;

    TmpPushMultipleBaitGooseToCatScript(const ServerState & state) :
            Script("TmpPushMultipleBaitGooseToLongScript", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
                   {ObserveType::Absolute, 0, {366.774475, 2669.538818, 239.860245}, {16.486465, -46.266056}}) { };

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard) {
            Blackboard & blackboard = *tree.blackboard;
            Script::initialize(tree, state);
            set<string> baiter0ValidLocations{"ShortStairs"}, baiter1ValidLocations{"ExtendedA", ""};
            Node::Ptr setupCommands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard, name),
                                                         make_unique<movement::WaitNode>(blackboard, 1.0),
                                                         make_unique<SpecDynamic>(blackboard, neededBots, observeSettings),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{neededBots[0].id, neededBots[1].id, neededBots[2].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({471.375, 2112.14, 96.02}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard,neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({440.58, 2259.15, 95.93}), Vec2({-1.760050, -105.049713})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[1].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({420.199219, 2377.000000, 96.528168}), Vec2({-0.659997, 5.090078})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, neededBots[2].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ForceOrderNode>(blackboard, "ForceTCat", vector{neededBots[0].id, neededBots[1].id, neededBots[2].id}, testAToCatWaypoints, addedOrderId),
                                                         make_unique<ForceEntryIndexNode>(blackboard, "ForcePusherBaiter",
                                                                                          vector{neededBots[0].id, neededBots[1].id, neededBots[2].id},
                                                                                          vector{0, 1, 2}),
                                                         make_unique<movement::WaitNode>(blackboard, 20.)),
                                                 "tmp");
            commands = make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                        std::move(setupCommands),
                        make_unique<DisableActionsNode>(blackboard, "DisableCondition",
                                                        vector{neededBots[0].id, neededBots[1].id, neededBots[2].id})),
                  "TmpPushMultiple"
            );
        }
    }
};
*/

#endif //CSKNOW_TEAMWORK_H
