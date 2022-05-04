//
// Created by durst on 5/3/22.
//

#include "bots/behavior_tree/priority/follow_order_node.h"

namespace follow {
    NodeState SoloTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const Order & curOrder = blackboard.orders[blackboard.playerToOrder[state.csgoIdToCSKnowId[treeThinker.csgoId]]];

        for (const auto & waypoint : curOrder.waypoints) {
            // if going to
            if (waypoint.waypointType == WaypointType::NavPlace) {
                if (treeThinker.placesInOrderVisited.find(waypoint.placeName) != treeThinker.placesInOrderVisited.end()) {

                }
            }
        }
    }
}
