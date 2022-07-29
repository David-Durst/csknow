//
// Created by steam on 6/30/22.
//

#ifndef CSKNOW_BLACKBOARD_MANAGEMENT_H
#define CSKNOW_BLACKBOARD_MANAGEMENT_H

#include "bots/testing/script.h"
#include "bots/behavior_tree/pathing_node.h"
#include "bots/analysis/save_nav_overlay.h"

const static Waypoints testCatToAWaypoints = {
        {WaypointType::NavPlace, "Catwalk"},
        {WaypointType::NavPlace, "ShortStairs"},
        {WaypointType::NavPlace, "ExtendedA"},
        {WaypointType::NavPlace, "BombsiteA"}
};

const static Waypoints testAToCatWaypoints = {
        {WaypointType::NavPlace, "BombsiteA"},
        {WaypointType::NavPlace, "ExtendedA"},
        {WaypointType::NavPlace, "ShortStairs"},
        {WaypointType::NavPlace, "Catwalk"}
};

class ForceOrderNode : public Node {
    vector<CSGOId> targetIds;
    Waypoints waypoints;
    set<AreaId> areaIdsToRemove;
    OrderId & orderId;
public:
    ForceOrderNode(Blackboard & blackboard, string name, vector<CSGOId> targetIds, Waypoints waypoints, OrderId & orderId) :
            Node(blackboard, name), targetIds(targetIds), waypoints(waypoints), orderId(orderId) { };
    ForceOrderNode(Blackboard & blackboard, string name, vector<CSGOId> targetIds, Waypoints waypoints, set<AreaId> areaIdsToRemove, OrderId & orderId) :
            Node(blackboard, name), targetIds(targetIds), waypoints(waypoints), areaIdsToRemove(areaIdsToRemove), orderId(orderId) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        //vector<string> pathPlace = { "Catwalk", "ShortStairs", "ExtendedA", "BombsiteA" };
        if (targetIds.empty()) {
            throw std::runtime_error("need at least one target to force order");
        }
        TeamId team = state.getClient(targetIds[0]).team;
        for (size_t i = 1; i < targetIds.size(); i++) {
            if (team != state.getClient(targetIds[i]).team) {
                throw std::runtime_error("all targets of new order must have same team");
            }
        }
        orderId = blackboard.strategy.addOrder(team, {waypoints}, blackboard.navFile,
                                               blackboard.reachability, blackboard.visPoints, blackboard.distanceToPlaces);
        for (const auto & targetId : targetIds) {
            blackboard.strategy.assignPlayerToOrder(targetId, orderId);
            blackboard.playerToPriority.erase(targetId);
        }
        if (!areaIdsToRemove.empty()) {
            blackboard.navFile.remove_incoming_edges_to_areas(areaIdsToRemove);
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
};

class ForceEntryIndexNode : public Node {
    vector<CSGOId> targetIds;
    vector<int32_t> entryIndices;
public:
    ForceEntryIndexNode(Blackboard & blackboard, string name, vector<CSGOId> targetIds, vector<int32_t> entryIndices) :
            Node(blackboard, name), targetIds(targetIds), entryIndices(entryIndices) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        for (size_t i = 0; i < targetIds.size(); i++) {
            blackboard.strategy.playerToEntryIndex[targetIds[i]] = entryIndices[i];
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
};

class ForceHoldIndexNode : public Node {
    vector<CSGOId> targetIds;
    vector<int> holdIndices;
    OrderId orderId;
public:
    ForceHoldIndexNode(Blackboard & blackboard, string name, vector<CSGOId> targetIds, vector<int> holdIndices, OrderId orderId) :
            Node(blackboard, name), targetIds(targetIds), holdIndices(holdIndices), orderId(orderId) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        for (size_t i = 0; i < targetIds.size(); i++) {
            // intentional int to size_t mismatch so easier to create test vectors, not gonna have more then 1000 waypoints anyway
            blackboard.strategy.assignPlayerToHoldIndex(targetIds[i], orderId, holdIndices[i]);
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
            const nav_mesh::nav_area & curArea =
                    blackboard.navFile.get_nearest_area_by_position(vec3Conv(state.getClient(targetIds[i]).getFootPosForPlayer()));
            if (dangerAreas[curDangerAreaIndex] || curDangerAreaIndex == blackboard.navFile.m_area_ids_to_indices[curArea.get_id()]) {
                playerNodeState[treeThinker.csgoId] = NodeState::Failure;
                return playerNodeState[treeThinker.csgoId];
            }
            dangerAreas[curDangerAreaIndex] = true;
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
};

class SavePossibleVisibleOverlays : public Node {
    vector<CSGOId> possibleAreasTargetIds;
    bool visibleOverlay;
    CSKnowTime oldTime = std::chrono::system_clock::from_time_t(0);
public:
    SavePossibleVisibleOverlays(Blackboard & blackboard, vector<CSGOId> possibleAreasTargetIds,
                                    bool visibleOverlay, string name = "SavePossibleVisibleOverlays") :
            Node(blackboard, name), possibleAreasTargetIds(possibleAreasTargetIds), visibleOverlay(visibleOverlay) { };
    virtual NodeState exec(const ServerState & state, TreeThinker &treeThinker) override {
        vector<AreaBits> overlays;
        CSKnowTime curTime = std::chrono::system_clock::now();
        //std::cout << "time since last save " << state.getSecondsBetweenTimes(oldTime, curTime) << std::endl;
        oldTime = curTime;
        if (!possibleAreasTargetIds.empty()) {
            map<CSGOId, AreaBits> playerToOverlay;
            for (CSGOId targetId : possibleAreasTargetIds) {
                overlays.push_back(blackboard.possibleNavAreas.getPossibleAreas(targetId));
            }
        }
        if (visibleOverlay) {
            map<CSGOId, AreaBits> teamToOverlay;
            overlays.push_back(blackboard.getVisibleAreasByTeam(state, ENGINE_TEAM_T));
            overlays.push_back(blackboard.getVisibleAreasByTeam(state, ENGINE_TEAM_CT));
        }
        blackboard.navFileOverlay.save(state, overlays);
        playerNodeState[treeThinker.csgoId] = NodeState::Running;
        return playerNodeState[treeThinker.csgoId];
    }
};

#endif //CSKNOW_BLACKBOARD_MANAGEMENT_H
