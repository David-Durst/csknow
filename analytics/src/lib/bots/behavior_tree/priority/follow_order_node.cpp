//
// Created by durst on 5/3/22.
//

#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"

namespace follow {
    Priority & getOrInitPriority(Blackboard & blackboard, TreeThinker & treeThinker) {
        if (blackboard.playerToPriority.find(treeThinker.csgoId) == blackboard.playerToPriority.end()) {
            blackboard.playerToPriority[treeThinker.csgoId] = {};
            blackboard.playerToPriority[treeThinker.csgoId].targetPlayer.firstTargetFrame = INVALID_ID;
        }
        return
    }


    NodeState PushTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const Order & curOrder = blackboard.orders[blackboard.playerToOrder[state.csgoIdToCSKnowId[treeThinker.csgoId]]];
        // default values are set to invalid where necessary, so this is fine
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];

        if (this->nodeState != NodeState::Running) {
            treeThinker.orderWaypointIndex++;
            // done with waypoints if index is over max, then just return success until getting new order
            if (treeThinker.orderWaypointIndex < curOrder.waypoints.size()) {
                moveToWaypoint(*this, state, treeThinker, curOrder, curPriority);
                this->nodeState = NodeState::Running;
            }
            else {
                this->nodeState = NodeState::Success;
            }
        }
        else {
            string curPlace = blackboard.getPlayerPlace(state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer());
            finishWaypoint(*this, state, treeThinker, curOrder, curPriority, curPlace);
        }

        return this->nodeState;
    }

    NodeState BaitTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const Order & curOrder = blackboard.orders[blackboard.playerToOrder[state.csgoIdToCSKnowId[treeThinker.csgoId]]];
        // default values are set to invalid where necessary, so this is fine
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];

        if (this->nodeState != NodeState::Running) {
            treeThinker.orderWaypointIndex++;
            // done with waypoints if index is over max, then just return success until getting new order
            if (treeThinker.orderWaypointIndex < curOrder.waypoints.size()) {
                moveToWaypoint(*this, state, treeThinker, curOrder, curPriority);
                this->nodeState = NodeState::Running;
            }
            else {
                this->nodeState = NodeState::Success;
            }
        }
        else {
            string curPlace = blackboard.getPlayerPlace(state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer());
            finishWaypoint(*this, state, treeThinker, curOrder, curPriority, curPlace);

            // if didn't finish, wait if some else on same order is closer
            if (this->nodeState == NodeState::Running) {
                Vec3 curPos = state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer();
                const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPos));
                const nav_mesh::nav_area & targetArea = blackboard.navFile.get_nearest_area_by_position(
                        vec3Conv(curPriority.targetPos));
                double curDistanceToGoal = blackboard.reachability.getDistance(curArea.get_id(), targetArea.get_id());
                for (const auto & followerId : curOrder.followers) {
                    if (followerId == treeThinker.csgoId) {
                        continue;
                    }
                    else {
                        Vec3 otherPos = state.clients[state.csgoIdToCSKnowId[followerId]].getFootPosForPlayer();
                        const nav_mesh::nav_area & otherArea = blackboard.navFile.get_nearest_area_by_position(
                                vec3Conv(otherPos));
                        double distanceDelta = blackboard.reachability.getDistance(otherArea.get_id(), targetArea.get_id()) - curDistanceToGoal;
                        if (distanceDelta > 30. & distanceDelta < 100.) {
                            curPriority.priorityMovement = PriorityMovement::Wait;
                            break;
                        }
                    }
                }
            }
        }

        return this->nodeState;
    }
}
