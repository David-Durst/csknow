//
// Created by steam on 7/31/22.
//
#include "bots/behavior_tree/priority/spacing_helpers.h"

NumAheadResult computeNumAhead(Blackboard & blackboard, const ServerState & state, const ServerState::Client & curClient) {
    NumAheadResult result{0, 0, std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max()};
    const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(
            {curClient.lastEyePosX, curClient.lastEyePosY, curClient.lastFootPosZ});
    const OrderId curOrderId = blackboard.strategy.getOrderIdForPlayer(curClient.csgoId);
    const Order & curOrder = blackboard.strategy.getOrderForPlayer(curClient.csgoId);
    const Waypoint & lastWaypoint = curOrder.waypoints.back();
    AreaId lastAreaId = blackboard.distanceToPlaces.getClosestArea(curArea.get_id(), lastWaypoint.placeName, blackboard.navFile);
    Vec3 curPos = curClient.getFootPosForPlayer();
    Vec3 targetPos = vec3tConv(blackboard.navFile.get_area_by_id_fast(lastAreaId).get_center());

    std::optional<vector<nav_mesh::PathNode>> curClientWaypoints =
            blackboard.navFile.find_path_detailed( vec3Conv(curPos), vec3Conv(targetPos));

    float curClientDistanceToTarget;
    // just return 0 if can't path for current client
    if (curClientWaypoints) {
        curClientDistanceToTarget = blackboard.navFile.compute_path_length_from_origin(vec3Conv(curPos), curClientWaypoints.value());
    }
    else {
        return result;
    }

    // find other order followers who are ahead, make sure num ahead matches number assigned to follow
    for (const auto & followerId : blackboard.strategy.getOrderFollowers(curOrderId)) {
        if (followerId == curClient.csgoId) {
            continue;
        }
        else {
            Vec3 otherPos = state.clients[state.csgoIdToCSKnowId[followerId]].getFootPosForPlayer();
            AreaId otherAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(otherPos)).get_id();
            AreaId otherLastAreaId = blackboard.distanceToPlaces.getClosestArea(otherAreaId, lastWaypoint.placeName, blackboard.navFile);
            Vec3 targetPos = vec3tConv(blackboard.navFile.get_area_by_id_fast(otherLastAreaId).get_center());

            std::optional<vector<nav_mesh::PathNode>> otherClientWaypoints =
                    blackboard.navFile.find_path_detailed(vec3Conv(otherPos), vec3Conv(targetPos));

            float otherClientDistanceToTarget;
            // quit if can't path for current client or finished
            if (otherClientWaypoints) {
                otherClientDistanceToTarget = blackboard.navFile.compute_path_length_from_origin(vec3Conv(otherPos), otherClientWaypoints.value());
            }
            else {
                continue;
            }
            // other person ahead if you are more than BAIT_DISTANCE behind them
            double distanceInFront = curClientDistanceToTarget - otherClientDistanceToTarget;
            // allow tolerance since can have different paths
            if (distanceInFront > 0) {
                result.numAhead++;
            }
            else {
                result.numBehind++;
            }
            // only count them as too close if not finished
            if (blackboard.strategy.playersFinishedStrategy.count(followerId) == 0) {
                if (distanceInFront > 0) {
                    result.nearestInFront = std::min(result.nearestInFront, distanceInFront);
                }
                else {
                    result.nearestBehind = std::min(result.nearestBehind, -1 * distanceInFront);
                }
            }
        }
    }
    return result;
}
