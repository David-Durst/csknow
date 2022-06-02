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
        return blackboard.playerToPriority[treeThinker.csgoId];
    }


    NodeState PushTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const Order & curOrder = blackboard.orders[blackboard.playerToOrder[treeThinker.csgoId]];
        bool havePriority = blackboard.playerToPriority.find(treeThinker.csgoId) != blackboard.playerToPriority.end();

        // default values are set to invalid where necessary, so this is fine
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];

        // if no priority yet, move to first one
        if (!havePriority) {
            moveToWaypoint(*this, state, treeThinker, curOrder, curPriority);
        }

        string curPlace = blackboard.getPlayerPlace(state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer());
        bool finishedWaypoint = finishWaypoint(*this, state, treeThinker, curOrder, curPriority, curPlace);

        bool finishedAndDone = false;
        // if finished waypoint,
        if (finishedWaypoint) {
            // need to pick a new path on priority change
            blackboard.playerToPath.erase(treeThinker.csgoId);
            // increment counter and move to next waypoint if possible
            if (treeThinker.orderWaypointIndex < curOrder.waypoints.size() - 1) {
                treeThinker.orderWaypointIndex++;
                moveToWaypoint(*this, state, treeThinker, curOrder, curPriority);
            }
            else {
                // stop moving if nothing to do
                finishedAndDone = true;
            }
        }


        const Action &priorAction = blackboard.lastPlayerToAction[treeThinker.csgoId];
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);

        // fail pushing if stuck
        if (havePriority && !finishedWaypoint && priorAction.moving() && computeMagnitude(curClient.getVelocity()) < MOVING_THRESHOLD) {
            curPriority.stuckTicks++;
        }
        else {
            curPriority.stuckTicks = 0;
        }

        if (curPriority.stuckTicks > STUCK_TICKS_THRESHOLD) {
            curPriority.stuckTicks = 0;
            curPriority.stuck = true;
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }
        else {
            playerNodeState[treeThinker.csgoId] = finishedAndDone ? NodeState::Success : NodeState::Running;
        }

        return playerNodeState[treeThinker.csgoId];
    }

    NodeState StuckTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const Order & curOrder = blackboard.orders[blackboard.playerToOrder[treeThinker.csgoId]];

        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        curPriority.targetAreaId = getRandomAreaInNextPlace(state, curOrder.waypoints[treeThinker.orderWaypointIndex].placeName);
        curPriority.targetPos = vec3tConv(blackboard.navFile.get_area_by_id_fast(curPriority.targetAreaId).get_center());
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

    // TODO: UNIFY CODE WITH PUSH SO BOTH USE SAME SKELETON
    NodeState BaitTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const Order & curOrder = blackboard.orders[blackboard.playerToOrder[treeThinker.csgoId]];
        // default values are set to invalid where necessary, so this is fine
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];

        // if no priority yet or get to current waypoint, setup next one
        if (playerNodeState.find(treeThinker.csgoId) == playerNodeState.end() ||
            playerNodeState[treeThinker.csgoId] != NodeState::Running) {
            treeThinker.orderWaypointIndex++;
            // done with waypoints if index is over max, then just return success until getting new order
            if (treeThinker.orderWaypointIndex < curOrder.waypoints.size()) {
                moveToWaypoint(*this, state, treeThinker, curOrder, curPriority);
                playerNodeState[treeThinker.csgoId] = NodeState::Running;
            }
            else {
                playerNodeState[treeThinker.csgoId] = NodeState::Success;
            }
        }
        else {
            string curPlace = blackboard.getPlayerPlace(state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer());
            finishWaypoint(*this, state, treeThinker, curOrder, curPriority, curPlace);

            // baiting logic - find anyone with same order who is closer and stop if they are too close
            Vec3 curPos = state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer();
            const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPos));
            const nav_mesh::nav_area & targetArea = blackboard.navFile.get_nearest_area_by_position(
                    vec3Conv(curPriority.targetPos));
            double curDistanceToGoal = blackboard.getDistance(curArea.get_id(), targetArea.get_id());
            for (const auto & followerId : curOrder.followers) {
                if (followerId == treeThinker.csgoId) {
                    continue;
                }
                else {
                    Vec3 otherPos = state.clients[state.csgoIdToCSKnowId[followerId]].getFootPosForPlayer();
                    const nav_mesh::nav_area & otherArea = blackboard.navFile.get_nearest_area_by_position(
                            vec3Conv(otherPos));
                    double distanceDelta = blackboard.getDistance(otherArea.get_id(), targetArea.get_id()) - curDistanceToGoal;
                    if (distanceDelta < 50.) {
                        break;
                    }
                }
            }
        }

        return playerNodeState[treeThinker.csgoId];
    }
}
