//
// Created by steam on 6/20/22.
//

#ifndef CSKNOW_BASIC_NAV_H
#define CSKNOW_BASIC_NAV_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/behavior_tree/tree.h"

class ForceTCatOrderNode : public Node {
    CSGOId targetId;
public:
    ForceTCatOrderNode(Blackboard & blackboard, CSGOId targetId) :
        Node(blackboard, "ForceTCatOrderNode"), targetId(targetId) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        vector<string> pathPlace = { "Catwalk", "ShortStairs", "ExtendedA", "BombsiteA" };
        vector<Waypoint> waypoints;
        for (const auto & p : pathPlace) {
            waypoints.push_back({WaypointType::NavPlace, p, INVALID_ID});
        }
        blackboard.orders.push_back({waypoints, {}, {}, {targetId}});
        for (const auto & client : state.clients) {
            blackboard.playerToOrder[client.csgoId] = blackboard.orders.size() - 1;
            blackboard.playerToTreeThinkers[client.csgoId].orderWaypointIndex = 0;
            blackboard.playerToPriority.erase(client.csgoId);
        }
        blackboard.navFile.remove_incoming_edges_to_areas({4048});
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return NodeState::Success;
    }
};

class JumpedBeforeCat : public Node {
    CSGOId targetId;
public:
    JumpedBeforeCat(Blackboard & blackboard, CSGOId targetId) :
            Node(blackboard, "ValidConditionNod"), targetId(targetId) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        const ServerState::Client & curClient = state.getClient(targetId);
        uint32_t curArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curClient.getFootPosForPlayer())).get_id();
        string curPlace = blackboard.navFile.get_place(blackboard.navFile.get_area_by_id_fast(curArea).m_place);
        // fail if get to cat before jumping on the boxes
        if (curPlace == "Catwalk") {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
            return NodeState::Failure;
        }
        else if (curArea == 8799) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
            return NodeState::Success;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
            return NodeState::Running;
        }
    }
};

class GooseToCatScript : public Script {
public:
    GooseToCatScript(const ServerState & state) :
        Script("GooseToLongScript", {{0, ENGINE_TEAM_T}}, {ObserveType::FirstPerson, 0}) { }

    virtual void initialize(Tree & tree, ServerState & state) override  {
        if (tree.newBlackboard)  {
            Blackboard & blackboard = *tree.blackboard;
            blackboard.neededBots = neededBots;
            blackboard.observeSettings = observeSettings;
            Script::initialize(tree, state);
            commands = make_unique<SequenceNode>(blackboard, Node::makeList(
                                                         make_unique<InitTestingRound>(blackboard),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SpecDynamic>(blackboard),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SlayAllBut>(blackboard, vector{blackboard.neededBots[0].id},state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<SetPos>(blackboard, Vec3({1071.936035, 2972.308837, 128.762023}), Vec2({2.903987, -95.587982})),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<Teleport>(blackboard, blackboard.neededBots[0].id, state),
                                                         make_unique<movement::WaitNode>(blackboard, 0.1),
                                                         make_unique<ForceTCatOrderNode>(blackboard, blackboard.neededBots[0].id),
                                                         make_unique<ParallelFirstNode>(blackboard, Node::makeList(
                                                                                                make_unique<JumpedBeforeCat>(blackboard, blackboard.neededBots[0].id),
                                                                                                make_unique<movement::WaitNode>(blackboard, 10, false)),
                                                                                        "GooseToLongCondition")),
                                                 "GooseToLongSequence");
        }
    }
};

#endif //CSKNOW_BASIC_NAV_H
