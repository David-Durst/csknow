//
// Created by durst on 5/4/22.
//

#include "bots/behavior_tree/priority/priority_helpers.h"

void moveToWaypoint(Node & node, const ServerState & state, TreeThinker & treeThinker,
                    const Order & curOrder, Priority & curPriority) {
    const Waypoint & waypoint = curOrder.waypoints[treeThinker.orderWaypointIndex];
    // if next area is a nav place, go there
    if (waypoint.waypointType == WaypointType::NavPlace) {
        curPriority.targetAreaId = node.getNearestAreaInNextPlace(state, treeThinker, waypoint.placeName);
        curPriority.targetPos = vec3tConv(node.blackboard.navFile.get_area_by_id_fast(curPriority.targetAreaId).get_center());
    }
    else if (waypoint.waypointType == WaypointType::Player) {
        curPriority.targetPos = state.getClient(waypoint.playerId).getFootPosForPlayer();
        curPriority.targetAreaId = node.blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPriority.targetPos)).get_id();
    }
    else if (waypoint.waypointType == WaypointType::C4) {
        curPriority.targetPos = state.getC4Pos();
        curPriority.targetAreaId = node.blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPriority.targetPos)).get_id();
    }
    curPriority.movementOptions = {true, false, false};
    curPriority.shootOptions = PriorityShootOptions::DontShoot;

}

bool finishWaypoint(Node & node, const ServerState & state, TreeThinker & treeThinker,
                    const Order & curOrder, Priority & curPriority, string curPlace) {
    // finished with current priority if
    // trying to reach place and got there
    if (curOrder.waypoints[treeThinker.orderWaypointIndex].waypointType == WaypointType::NavPlace) {
        if (curOrder.waypoints[treeThinker.orderWaypointIndex].placeName == curPlace) {
            node.playerNodeState[treeThinker.csgoId] = NodeState::Success;
            return true;
        }
    }
    // target player died
    else if (curOrder.waypoints[treeThinker.orderWaypointIndex].waypointType == WaypointType::Player) {
        if (!state.clients[state.csgoIdToCSKnowId[curOrder.waypoints[treeThinker.orderWaypointIndex].playerId]].isAlive) {
            node.playerNodeState[treeThinker.csgoId] = NodeState::Success;
            return true;
        }
    }
    // c4 doesn't finish
    return false;
}
