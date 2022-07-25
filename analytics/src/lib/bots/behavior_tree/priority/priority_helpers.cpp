//
// Created by durst on 5/4/22.
//

#include "bots/behavior_tree/priority/priority_helpers.h"

void moveToWaypoint(Node & node, const ServerState & state, TreeThinker & treeThinker,
                    const Order & curOrder, Priority & curPriority, const Strategy & strategy) {
    const Waypoint & waypoint = curOrder.waypoints[strategy.playerToWaypointIndex.find(treeThinker.csgoId)->second];
    // if next area is a nav place, go there
    if (waypoint.type == WaypointType::NavPlace) {
        curPriority.targetAreaId = node.getNearestAreaInNextPlace(state, treeThinker, waypoint.placeName);
        curPriority.targetPos = vec3tConv(node.blackboard.navFile.get_area_by_id_fast(curPriority.targetAreaId).get_center());
    }
    else if (waypoint.type == WaypointType::Player) {
        curPriority.targetPos = state.getClient(waypoint.playerId).getFootPosForPlayer();
        curPriority.targetAreaId = node.blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPriority.targetPos)).get_id();
    }
    else if (waypoint.type == WaypointType::C4) {
        curPriority.targetPos = state.getC4Pos();
        curPriority.targetAreaId = node.blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPriority.targetPos)).get_id();
    }
    curPriority.stuck = false;
    curPriority.stuckTicks = 0;
}

bool finishWaypoint(const ServerState & state, int64_t waypointIndex,
                    const Order & curOrder, Priority & curPriority, string curPlace) {
    // finished with current priority if
    // trying to reach place and got there
    if (curOrder.waypoints[waypointIndex].type == WaypointType::NavPlace) {
        if (curOrder.waypoints[waypointIndex].placeName == curPlace) {
            return true;
        }
    }
    // target player died
    else if (curOrder.waypoints[waypointIndex].type == WaypointType::Player) {
        if (!state.clients[state.csgoIdToCSKnowId[curOrder.waypoints[waypointIndex].playerId]].isAlive) {
            return true;
        }
    }
    // c4 doesn't finish
    return false;
}

int64_t getMaxFinishedWaypoint(const ServerState & state, const Order & curOrder, Priority & curPriority, string curPlace) {
    int64_t maxFinishedWaypointIndex = -1;
    for (size_t i = 0; i < curOrder.waypoints.size(); i++) {
        if (finishWaypoint(state, i, curOrder, curPriority, curPlace)) {
            maxFinishedWaypointIndex = i;
        }
    }
    return maxFinishedWaypointIndex;
}
