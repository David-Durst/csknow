//
// Created by steam on 7/11/22.
//

#include "bots/behavior_tree/global/communicate_node.h"
#include <array>
using std::array;

namespace communicate {
    struct AssignedAreas {
        AreaBits tAssignedAreas, ctAssignedAreas;

        AreaBits & getTeamAssignedAreas(const ServerState &state, TreeThinker &treeThinker) {
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
        size_t areaIndex;
        AreaId areaId;
        double distance;
    };

    NodeState PrioritizeDangerAreasNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        AssignedAreas assignedAreas;

        // clear out assignments from prior frame
        blackboard.playerToDangerAreaId.clear();

        AreaBits tVisibleAreas = blackboard.getVisibleAreasByTeam(state, ENGINE_TEAM_T),
                ctVisibleAreas = blackboard.getVisibleAreasByTeam(state, ENGINE_TEAM_CT);

        for (const auto & client : state.clients) {
            if (client.isAlive && client.isBot) {
                AreaBits teamAssignedAreas = assignedAreas.getTeamAssignedAreas(state, treeThinker);

                const nav_mesh::nav_area & curArea =
                        blackboard.navFile.get_nearest_area_by_position(vec3Conv(client.getFootPosForPlayer()));

                // filter for cover edge - visible, but adjacent to behind cover area
                vector<CoverEdge> coverEdges;
                for (size_t i = 0; i < blackboard.navFile.m_areas.size(); i++) {
                    if ((client.team == ENGINE_TEAM_T && tVisibleAreas[i]) || (client.team == ENGINE_TEAM_CT && ctVisibleAreas[i])) {
                        for (size_t j = 0; j < blackboard.navFile.connections_area_length[i]; j++) {
                            size_t conAreaIndex = blackboard.navFile.connections[blackboard.navFile.connections_area_start[i] + j];
                            if ((client.team == ENGINE_TEAM_T && !tVisibleAreas[conAreaIndex]) || (client.team == ENGINE_TEAM_CT && !ctVisibleAreas[conAreaIndex])) {
                                // distance is distance to possible enemy locations
                                double minDistance = std::numeric_limits<double>::max();
                                for (const auto & possibleAreaIndex : blackboard.possibleNavAreas.getEnemiesPossiblePositions(state, treeThinker.csgoId)) {
                                    double tmpDistance = blackboard.reachability.getDistance(possibleAreaIndex, i);
                                    if (tmpDistance < minDistance) {
                                        minDistance = tmpDistance;
                                    }
                                }
                                // if there are no enemies, then no need to worry about cover edges
                                if (minDistance != std::numeric_limits<double>::max()) {
                                    coverEdges.push_back({i, blackboard.navFile.m_areas[i].get_id(), minDistance});
                                }
                                break;
                            }
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
                    if (!teamAssignedAreas[coverEdge.areaIndex]) {
                        blackboard.playerToDangerAreaId[client.csgoId] = coverEdge.areaId;
                    }
                }
                if (blackboard.playerToDangerAreaId.find(client.csgoId) == blackboard.playerToDangerAreaId.end()) {
                    blackboard.playerToDangerAreaId[client.csgoId] = coverEdges[0].areaId;
                }

                size_t srcAreaIndex = blackboard.navFile.m_area_ids_to_indices[blackboard.playerToDangerAreaId[client.csgoId]];
                // count all areas within distance to assigned one as also assigned
                for (size_t dstAreaIndex = 0; dstAreaIndex < blackboard.navFile.m_areas.size(); dstAreaIndex++) {
                    if (blackboard.reachability.getDistance(srcAreaIndex, dstAreaIndex) < WATCHED_DISTANCE) {
                        teamAssignedAreas[dstAreaIndex] = true;
                    }
                }
            }
        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
