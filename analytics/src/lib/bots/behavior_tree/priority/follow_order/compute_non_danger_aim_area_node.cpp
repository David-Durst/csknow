//
// Created by steam on 8/2/22.
//
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"

namespace follow {
    NodeState ComputeNonDangerAimAreaNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        const Order & curOrder = blackboard.strategy.getOrderForPlayer(treeThinker.csgoId);
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        AreaId curAreaId = blackboard.navFile.get_nearest_area_by_position(
                vec3Conv(state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer())).get_id();
        size_t lastWaypointIndex;
        if (curClient.team == ENGINE_TEAM_CT) {
            lastWaypointIndex = curOrder.waypoints.size() - 1;
        }
        else {
            lastWaypointIndex = curOrder.playerToHoldIndex.find(treeThinker.csgoId)->second;
        }
        const Waypoint & lastWaypoint = curOrder.waypoints[lastWaypointIndex];

        // set non danger aim area equal to visible area closest to destination if not already set by
        // computing hold nav area
        if (!curPriority.nonDangerAimArea) {
            AreaBits visibleAreas = blackboard.visPoints.getVisibilityRelativeToSrc(curAreaId);
            struct VisDistance {
                size_t areaIndex;
                double distance;
            };
            vector<VisDistance> visDistances;
            AreaId lastAreaId;
            // TODO: handle when last waypoint isn't a C4
            if (curClient.team == ENGINE_TEAM_CT) {
                lastAreaId = blackboard.distanceToPlaces.getMedianArea(curAreaId, lastWaypoint.placeName, blackboard.navFile);
            }
            else {
                // ok to just use target area as T is always just pathing to hold point
                lastAreaId = curPriority.targetAreaId;
            }
            size_t lastAreaIndex = blackboard.navFile.m_area_ids_to_indices[lastAreaId];
            for (size_t i = 0; i < visibleAreas.size(); i++) {
                if (visibleAreas[i]) {
                    double areaDistance = blackboard.reachability.getDistance(i, lastAreaIndex);
                    if (areaDistance != NOT_CLOSEST_DISTANCE) {
                        visDistances.push_back({i, areaDistance});
                    }
                }
            }

            std::sort(visDistances.begin(), visDistances.end(),
                      [](const VisDistance & a, const VisDistance & b) { return a.distance < b.distance; });
            curPriority.nonDangerAimArea = blackboard.navFile.m_areas[visDistances[0].areaIndex].get_id();
            curPriority.nonDangerAimAreaType = NonDangerAimAreaType::Path;
        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
