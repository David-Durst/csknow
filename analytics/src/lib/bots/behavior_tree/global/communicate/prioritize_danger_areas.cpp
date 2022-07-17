//
// Created by steam on 7/11/22.
//

#include "bots/behavior_tree/global/communicate_node.h"

namespace communicate {
    struct AssignedAreas {
        set<AreaId> tAssignedAreas, ctAssignedAreas;

        set<AreaId> & getTeamAssignedAreas(const ServerState &state, TreeThinker &treeThinker) {
            if (state.getClient(treeThinker.csgoId).team == ENGINE_TEAM_T) {
                return tAssignedAreas;
            }
            else {
                return ctAssignedAreas;
            }
        }
    };

    // cover edges are defined relative to another nav area
    struct CoverEdge {
        AreaId id;
        double distance;
    };

    NodeState PrioritizeDangerAreasNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        AssignedAreas assignedAreas;

        // clear out assignments from prior frame
        blackboard.playerToDangerAreaId.clear();

        for (const auto & client : state.clients) {
            if (client.isAlive && client.isBot) {
                set<AreaId> teamAssignedAreas = assignedAreas.getTeamAssignedAreas(state, treeThinker);

                const nav_mesh::nav_area & curArea =
                        blackboard.navFile.get_nearest_area_by_position(vec3Conv(client.getFootPosForPlayer()));

                // filter for cover edge - visible, but adjacent to behind cover area
                set<AreaId> visibleAreas = blackboard.visPoints.getAreasRelativeToSrc(curArea.get_id(), true);
                vector<CoverEdge> coverEdges;
                for (AreaId visibleAreaId : visibleAreas) {
                    size_t visibleAreaIndex = blackboard.navFile.m_area_ids_to_indices[visibleAreaId];
                    for (const auto & connection : blackboard.navFile.get_area_by_id_fast(visibleAreaId).get_connections()) {
                        if (visibleAreas.find(connection.id) == visibleAreas.end()) {
                            // distance is distance to possible enemy locations
                            double minDistance = std::numeric_limits<double>::max();
                            for (const auto & possibleAreaId : getEnemiesPossiblePositions(state, treeThinker.csgoId, blackboard.possibleNavAreas)) {
                                double tmpDistance = blackboard.reachability.getDistance(blackboard.navFile.m_area_ids_to_indices[possibleAreaId], visibleAreaIndex);
                                if (tmpDistance < minDistance) {
                                    minDistance = tmpDistance;
                                }
                            }
                            // if there are no enemies, then no need to worry about cover edges
                            if (minDistance == std::numeric_limits<double>::max()) {
                                coverEdges.push_back({visibleAreaId, minDistance});
                            }
                            break;
                        }
                    }
                }

                // if no cover edges, no places to pick from
                if (coverEdges.empty()) {
                    blackboard.playerToDangerAreaId.erase(client.csgoId);
                    continue;
                }

                // sort cover edges by distance to player (nearest first)
                std::sort(coverEdges.begin(), coverEdges.end(),
                          [](const CoverEdge & a, const CoverEdge & b) { return a.distance < b.distance; });

                // find first non-assigned cover edge
                // if all already assigned, take the closest one
                for (const auto & coverEdge : coverEdges) {
                    if (teamAssignedAreas.find(coverEdge.id) == teamAssignedAreas.end()) {
                        blackboard.playerToDangerAreaId[client.csgoId] = coverEdge.id;
                    }
                }
                if (blackboard.playerToDangerAreaId.find(client.csgoId) == blackboard.playerToDangerAreaId.end()) {
                    blackboard.playerToDangerAreaId[client.csgoId] = coverEdges[0].id;
                }

                size_t srcAreaIndex = blackboard.navFile.m_area_ids_to_indices[blackboard.playerToDangerAreaId[client.csgoId]];
                // count all areas within distance to assigned one as also assigned
                for (size_t dstAreaIndex = 0; dstAreaIndex < blackboard.navFile.m_areas.size(); dstAreaIndex++) {
                    if (blackboard.reachability.getDistance(srcAreaIndex, dstAreaIndex) < WATCHED_DISTANCE) {
                        teamAssignedAreas.insert(blackboard.navFile.m_areas[dstAreaIndex].get_id());
                    }
                }
            }
        }

        if (blackboard.inTest) {
            for (const auto & client : state.clients) {
                if (blackboard.possibleNavAreas[client.csgoId].find(4182) != blackboard.possibleNavAreas[client.csgoId].end() && client.team == ENGINE_TEAM_T) {
                    std::cout << 4182 << std::endl;
                }
            }
        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
