//
// Created by durst on 6/9/22.
//
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"

namespace follow {

    NodeState BaitMovementNode::exec(const ServerState & state, TreeThinker &treeThinker) {
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        curPriority.moveOptions.move = false;
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

    bool BaitConditionNode::valid(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        const Order & curOrder = blackboard.orders[blackboard.playerToOrder[treeThinker.csgoId]];
        const Waypoint & lastWaypoint = curOrder.waypoints.back();
        uint32_t lastAreaId = getNearestAreaInNextPlace(state, treeThinker, lastWaypoint.placeName);
        Vec3 targetPos = vec3tConv(blackboard.navFile.get_area_by_id_fast(lastAreaId).get_center());

        std::optional<vector<nav_mesh::PathNode>> curClientWaypoints =
                blackboard.navFile.find_path_detailed(
                        vec3Conv(curClient.getFootPosForPlayer()), vec3Conv(targetPos));

        float curClientDistanceToTarget;
        // quit if can't path for current client
        if (curClientWaypoints) {
            curClientDistanceToTarget = blackboard.navFile.compute_path_length(curClientWaypoints.value());
        }
        else {
            return false;
        }

        // find other order followers who are ahead, make sure num ahead matches number assigned to follow
        int numAhead = 0;
        for (const auto & followerId : curOrder.followers) {
            if (followerId == treeThinker.csgoId) {
                continue;
            }
            else {
                Vec3 otherPos = state.clients[state.csgoIdToCSKnowId[followerId]].getFootPosForPlayer();

                std::optional<vector<nav_mesh::PathNode>> otherClientWaypoints =
                        blackboard.navFile.find_path_detailed(vec3Conv(otherPos), vec3Conv(targetPos));

                float otherClientDistanceToTarget;
                // quit if can't path for current client
                if (otherClientWaypoints) {
                    otherClientDistanceToTarget = blackboard.navFile.compute_path_length(otherClientWaypoints.value());
                }
                else {
                    continue;
                }
                // other person ahead if you are more than BAIT_DISTANCE behind them
                if (curClientDistanceToTarget - otherClientDistanceToTarget > BAIT_DISTANCE) {
                    numAhead++;
                }
            }
        }
        // stop if too few people are ahead
        return numAhead < blackboard.playerToEntryIndex[treeThinker.csgoId];
    }
}