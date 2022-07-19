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
    set<AreaId> areaIdsToRemove;
public:
    ForceOrderNode(Blackboard & blackboard, string name, vector<CSGOId> targetIds, vector<string> pathPlaces) :
            Node(blackboard, name), targetIds(targetIds), pathPlaces(pathPlaces) { };
    ForceOrderNode(Blackboard & blackboard, string name, vector<CSGOId> targetIds, vector<string> pathPlaces, set<AreaId> areaIdsToRemove) :
            Node(blackboard, name), targetIds(targetIds), pathPlaces(pathPlaces), areaIdsToRemove(areaIdsToRemove) { };
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
        if (!areaIdsToRemove.empty()) {
            blackboard.navFile.remove_incoming_edges_to_areas(areaIdsToRemove);
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
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
        return playerNodeState[treeThinker.csgoId];
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
        playerNodeState[treeThinker.csgoId] = NodeState::Running;
        return playerNodeState[treeThinker.csgoId];
    }
};

class ClearMemoryCommunicationDangerNode : public Node {
public:
    ClearMemoryCommunicationDangerNode(Blackboard & blackboard) :
            Node(blackboard, "ClearMemoryCommunicationDanger") { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        blackboard.tMemory.positions.clear();
        blackboard.ctMemory.positions.clear();
        for (auto & [_, memory] : blackboard.playerToMemory) {
            memory.positions.clear();
        }
        blackboard.resetPossibleNavAreas = true;
        blackboard.inTest = true;
        blackboard.playerToDangerAreaId.clear();
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
};

class CheckPossibleLocationsNode : public Node {
    vector<CSGOId> targetIds;
    vector<vector<AreaId>> requiredPossibleAreas;
    vector<vector<AreaId>> requiredNotPossibleAreas;
public:
    CheckPossibleLocationsNode(Blackboard & blackboard, vector<CSGOId> targetIds,
                               vector<vector<AreaId>> requiredPossibleAreas, vector<vector<AreaId>> requiredNotPossibleAreas, string name = "CheckPossibleLocations") :
            Node(blackboard, name), targetIds(targetIds),
            requiredPossibleAreas(requiredPossibleAreas), requiredNotPossibleAreas(requiredNotPossibleAreas) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        for (size_t i = 0; i < targetIds.size(); i++) {
            // check the areas that have to be present
            for (size_t j = 0; j < requiredPossibleAreas[i].size(); j++) {
                if (!blackboard.possibleNavAreas.get(targetIds[i], requiredPossibleAreas[i][j])){
                    playerNodeState[treeThinker.csgoId] = NodeState::Failure;
                    return playerNodeState[treeThinker.csgoId];
                }
            }

            // check the areas that have to not be present
            for (size_t j = 0; j < requiredNotPossibleAreas[i].size(); j++) {
                if (blackboard.possibleNavAreas.get(targetIds[i], requiredNotPossibleAreas[i][j])) {
                    playerNodeState[treeThinker.csgoId] = NodeState::Failure;
                    return playerNodeState[treeThinker.csgoId];
                }
            }
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
};

class RequireDifferentDangerAreasNode : public Node {
    vector<CSGOId> targetIds;
public:
    RequireDifferentDangerAreasNode(Blackboard & blackboard, vector<CSGOId> targetIds, string name = "RequireDifferentDangerAreas") :
            Node(blackboard, name), targetIds(targetIds) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        AreaBits dangerAreas;
        for (size_t i = 0; i < targetIds.size(); i++) {
            size_t curDangerAreaIndex = blackboard.navFile.m_area_ids_to_indices.find(blackboard.playerToDangerAreaId[targetIds[i]])->second;
            if (dangerAreas[curDangerAreaIndex]) {
                playerNodeState[treeThinker.csgoId] = NodeState::Failure;
                return playerNodeState[treeThinker.csgoId];
            }
            dangerAreas[curDangerAreaIndex] = true;
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
};

#endif //CSKNOW_BLACKBOARD_MANAGEMENT_H
