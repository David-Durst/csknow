//
// Created by steam on 8/2/22.
//
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"

namespace follow {
    NodeState ComputeNonDangerAimAreaNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const Order & curOrder = blackboard.strategy.getOrderForPlayer(treeThinker.csgoId);
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        AreaId curAreaId = blackboard.navFile.get_nearest_area_by_position(
                vec3Conv(state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer())).get_id();
        const Waypoint & lastWaypoint = curOrder.waypoints.back();

        // set non danger aim area equal to visible area closest to destination if not already set by
        // computing hold nav area
        if (!curPriority.nonDangerAimArea) {
            AreaBits visibleAreas = blackboard.visPoints.getVisibilityRelativeToSrc(curAreaId);
            struct VisDistance {
                size_t areaIndex;
                double distance;
            };
            vector<VisDistance> visDistances;
            AreaId lastAreaId = blackboard.distanceToPlaces.getMedianArea(curAreaId, lastWaypoint.placeName, blackboard.navFile);
            size_t lastAreaIndex = blackboard.navFile.m_area_ids_to_indices[lastAreaId];
            for (size_t i = 0; i < visibleAreas.size(); i++) {
                if (visibleAreas[i]) {
                    visDistances.push_back({i, blackboard.reachability.getDistance(i, lastAreaIndex)});
                }
            }

            std::sort(visDistances.begin(), visDistances.end(),
                      [](const VisDistance & a, const VisDistance & b) { return a.distance < b.distance; });
            curPriority.nonDangerAimArea = blackboard.navFile.m_areas[visDistances[0].areaIndex].get_id();

        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
