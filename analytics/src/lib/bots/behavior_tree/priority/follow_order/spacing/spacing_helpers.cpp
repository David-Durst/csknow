//
// Created by steam on 7/31/22.
//
#include "bots/behavior_tree/priority/spacing_helpers.h"

int computeNumAhead(Blackboard & blackboard, const ServerState & state, const ServerState::Client & curClient) {
    const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(
            {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ});
    const OrderId curOrderId = blackboard.strategy.getOrderIdForPlayer(curClient.csgoId);
    const Order & curOrder = blackboard.strategy.getOrderForPlayer(curClient.csgoId);
    const Waypoint & lastWaypoint = curOrder.waypoints.back();
    AreaId lastAreaId = blackboard.distanceToPlaces.getClosestArea(curArea.get_id(), lastWaypoint.placeName, blackboard.navFile);
    Vec3 targetPos = vec3tConv(blackboard.navFile.get_area_by_id_fast(lastAreaId).get_center());

    std::optional<vector<nav_mesh::PathNode>> curClientWaypoints =
            blackboard.navFile.find_path_detailed(
                    vec3Conv(curClient.getFootPosForPlayer()), vec3Conv(targetPos));

    float curClientDistanceToTarget;
    // just return 0 if can't path for current client
    if (curClientWaypoints) {
        curClientDistanceToTarget = blackboard.navFile.compute_path_length(curClientWaypoints.value());
    }
    else {
        return 0;
    }

    // find other order followers who are ahead, make sure num ahead matches number assigned to follow
    int numAhead = 0;
    for (const auto & followerId : blackboard.strategy.getOrderFollowers(curOrderId)) {
        if (followerId == curClient.csgoId) {
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
    return numAhead;
}
