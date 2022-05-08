//
// Created by durst on 5/4/22.
//

#include "bots/behavior_tree/priority/priority_helpers.h"

void moveToWaypoint(Node & node, const ServerState & state, TreeThinker & treeThinker,
                    const Order & curOrder, Priority & curPriority) {
    const Waypoint & waypoint = curOrder.waypoints[treeThinker.orderWaypointIndex];
    // if next area is a nav place, go there
    if (waypoint.waypointType == WaypointType::NavPlace) {
        curPriority.priorityType = PriorityType::NavArea;
        curPriority.areaId = node.getNearestAreaInNextPlace(state, treeThinker, waypoint.placeName);
    }
    else if (waypoint.waypointType == WaypointType::Player) {
        curPriority.priorityType = PriorityType::Player;
        curPriority.playerId = waypoint.playerId;
    }
    else if (waypoint.waypointType == WaypointType::C4) {
        curPriority.priorityType = PriorityType::C4;
    }
    curPriority.movementOptions = {true, false, false};
    curPriority.shootOptions = PriorityShootOptions::DontShoot;

}

void finishWaypoint(Node & node, const ServerState & state, TreeThinker & treeThinker,
                    const Order & curOrder, Priority & curPriority, string curPlace) {
    // finished with current priority if
    // trying to reach place and got there
    if (curOrder.waypoints[treeThinker.orderWaypointIndex].waypointType == WaypointType::NavPlace) {
        if (curOrder.waypoints[treeThinker.orderWaypointIndex].placeName == curPlace) {
            node.nodeState = NodeState::Success;
        }
    }
        // target player died
    else if (curOrder.waypoints[treeThinker.orderWaypointIndex].waypointType == WaypointType::Player) {
        if (!state.clients[state.csgoIdToCSKnowId[curOrder.waypoints[treeThinker.orderWaypointIndex].playerId]].isAlive) {
            node.nodeState = NodeState::Success;
        }
    }
    // c4 doesn't finish
}
