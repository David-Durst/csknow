//
// Created by durst on 5/4/22.
//

#include "bots/behavior_tree/priority/priority_helpers.h"

NearestArea getNearestAreaInWaypoint(const Blackboard & blackboard, const ServerState & state,
                                     const nav_mesh::nav_area & curArea, const Waypoint & waypoint) {
    NearestArea result;
    // if next area is a nav place, go there
    if (waypoint.type == WaypointType::NavPlace) {
        result.targetAreaId = blackboard.distanceToPlaces.getMedianArea(curArea.get_id(), waypoint.placeName, blackboard.navFile);
        result.targetPos = vec3tConv(blackboard.navFile.get_area_by_id_fast(result.targetAreaId).get_center());
    }
    else if (waypoint.type == WaypointType::NavAreas) {
        double minDistance = std::numeric_limits<double>::max();
        AreaId minAreaId = INVALID_ID;
        for (const auto & dstAreaId : waypoint.areaIds) {
            if (blackboard.reachability.getDistance(curArea.get_id(), dstAreaId, blackboard.navFile) < minDistance) {
                minAreaId = dstAreaId;
            }
        }
        result.targetAreaId = minAreaId;
        result.targetPos = vec3tConv(blackboard.navFile.get_area_by_id_fast(result.targetAreaId).get_center());
    }
    else if (waypoint.type == WaypointType::C4) {
        result.targetPos = state.getC4Pos();
        result.targetAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(result.targetPos)).get_id();
    }
    return result;
}

void moveToWaypoint(const Blackboard & blackboard, const ServerState & state, TreeThinker & treeThinker,
                    const Order & curOrder, Priority & curPriority) {
    const Waypoint & waypoint = curOrder.waypoints[blackboard.strategy.playerToWaypointIndex.find(treeThinker.csgoId)->second];
    const nav_mesh::nav_area & curArea =
            blackboard.navFile.get_nearest_area_by_position(vec3Conv(state.getClient(treeThinker.csgoId).getFootPosForPlayer()));
    // if next area is a nav place, go there
    bool alreadyGoingToWaypoint = false;
    if (waypoint.type == WaypointType::NavPlace) {
        if (blackboard.navFile.m_area_ids_to_indices.find(curPriority.targetAreaId) != blackboard.navFile.m_area_ids_to_indices.end()) {
            size_t targetAreaIndex = blackboard.navFile.m_area_ids_to_indices.find(curPriority.targetAreaId)->second;
            string place = blackboard.navFile.get_place(blackboard.navFile.m_areas[targetAreaIndex].m_place);
            if (place == waypoint.placeName) {
                alreadyGoingToWaypoint = true;
            }
        }
    }
    else if (waypoint.type == WaypointType::NavAreas) {
        for (const auto & dstAreaId : waypoint.areaIds) {
            if (dstAreaId == curPriority.targetAreaId) {
                alreadyGoingToWaypoint = true;
                break;
            }
        }
    }
    else if (waypoint.type == WaypointType::C4) {
        // here for clarity, even though redundant
        alreadyGoingToWaypoint = false;
    }
    if (!alreadyGoingToWaypoint) {
        NearestArea nearestArea = getNearestAreaInWaypoint(blackboard, state, curArea, waypoint);
        curPriority.targetAreaId = nearestArea.targetAreaId;
        curPriority.targetPos = nearestArea.targetPos;
    }
}

bool finishWaypoint(const Blackboard & blackboard, const ServerState & state, int64_t waypointIndex,
                    const Order & curOrder, CSGOId playerId, string curPlace, AreaId curAreaId) {
    bool amDefuser = blackboard.isPlayerDefuser(playerId);
    // finished with current priority if
    // trying to reach place and got there
    if (curOrder.waypoints[waypointIndex].type == WaypointType::NavPlace ||
        (curOrder.waypoints[waypointIndex].type == WaypointType::C4 && !amDefuser)) {
        if (curOrder.waypoints[waypointIndex].placeName == curPlace) {
            return true;
        }
    }
    else if (curOrder.waypoints[waypointIndex].type == WaypointType::C4 && amDefuser) {
        if (computeDistance(state.getC4Pos(), state.getClient(playerId).getFootPosForPlayer()) < DEFUSE_DISTANCE) {
            return true;
        }
    }
    else if (curOrder.waypoints[waypointIndex].type == WaypointType::NavAreas) {
        for (const auto & areaId : curOrder.waypoints[waypointIndex].areaIds) {
            if (areaId == curAreaId) {
                return true;
            }
        }
    }
    return false;
}

int64_t getMaxFinishedWaypoint(const Blackboard & blackboard, const ServerState & state,
                               const Order & curOrder,
                               CSGOId playerId, string curPlace, AreaId curAreaId) {
    int64_t maxFinishedWaypointIndex = -1;
    for (size_t i = 0; i < curOrder.waypoints.size(); i++) {
        if (finishWaypoint(blackboard, state, i, curOrder, playerId, curPlace, curAreaId)) {
            maxFinishedWaypointIndex = i;
        }
    }
    return maxFinishedWaypointIndex;
}
