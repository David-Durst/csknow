//
// Created by steam on 6/30/22.
//

#ifndef CSKNOW_BLACKBOARD_MANAGEMENT_H
#define CSKNOW_BLACKBOARD_MANAGEMENT_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"

class ForceOrderNode : public Node {
    vector<CSGOId> targetIds;
    vector<string> pathPlaces;
public:
    ForceOrderNode(Blackboard & blackboard, string name, vector<CSGOId> targetIds, vector<string> pathPlaces) :
            Node(blackboard, name), targetIds(targetIds), pathPlaces(pathPlaces) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        //vector<string> pathPlace = { "Catwalk", "ShortStairs", "ExtendedA", "BombsiteA" };
        vector<Waypoint> waypoints;
        for (const auto & p : pathPlaces) {
            waypoints.push_back({WaypointType::NavPlace, p, INVALID_ID});
        }
        blackboard.orders.push_back({waypoints, {}, {}, targetIds});
        for (const auto & targetId : targetIds) {
            blackboard.playerToOrder[targetId] = blackboard.orders.size() - 1;
            blackboard.playerToTreeThinkers[targetId].orderWaypointIndex = 0;
            blackboard.playerToPriority.erase(targetId);
        }
        blackboard.navFile.remove_incoming_edges_to_areas({4048});
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return NodeState::Success;
    }
};

class ForceAggressionNode : public Node {
    vector<CSGOId> targetIds;
    vector<int32_t> pushIndices;
public:
    ForceAggressionNode(Blackboard & blackboard, string name, vector<CSGOId> targetIds, vector<int32_t> pushIndices) :
            Node(blackboard, name), targetIds(targetIds), pushIndices(pushIndices) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        for (size_t i = 0; i < targetIds.size(); i++) {
            blackboard.playerToPushOrder[targetIds[i]] = pushIndices[i];
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return NodeState::Success;
    }
};

class DisableActionsNode : public Node {
    vector<CSGOId> targetIds;
    bool disableMouse;
public:
    DisableActionsNode(Blackboard & blackboard, string name, vector<CSGOId> targetIds, bool disableMouse = true) :
            Node(blackboard, name), targetIds(targetIds), disableMouse(disableMouse) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        for (size_t i = 0; i < targetIds.size(); i++) {
            blackboard.playerToAction[targetIds[i]].buttons = 0;
            if (disableMouse) {
                blackboard.playerToAction[targetIds[i]].inputAngleDeltaPctX = 0.;
                blackboard.playerToAction[targetIds[i]].inputAngleDeltaPctY = 0.;
            }
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return NodeState::Running;
    }
};

#endif //CSKNOW_BLACKBOARD_MANAGEMENT_H
