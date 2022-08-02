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
            Script("PushBaitGooseToLongScript", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
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
                                                                                        "PushBaitGooseToLongCondition")),
                                                 "PushBaitGooseToLongSequence");
        }
    }
};

class PushMultipleBaitGooseToCatScript : public Script {
public:
    OrderId addedOrderId;

    PushMultipleBaitGooseToCatScript(const ServerState & state) :
            Script("PushMultipleBaitGooseToLongScript", {{0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}, {0, ENGINE_TEAM_CT}},
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
