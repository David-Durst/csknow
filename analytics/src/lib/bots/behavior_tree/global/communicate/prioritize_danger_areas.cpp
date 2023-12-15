//
// Created by steam on 7/11/22.
//

#include "bots/behavior_tree/global/communicate_node.h"
#include "geometry.h"
#include <array>
using std::array;

namespace communicate {
    struct AssignedAreas {
        AreaBits tAssignedAreas, ctAssignedAreas;

        AreaBits & getTeamAssignedAreas(const ServerState::Client & client) {
            if (client.team == ENGINE_TEAM_T) {
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
        double minTimeToEnemy;
        bool checkedRecently;
    };

    NodeState PrioritizeDangerAreasNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        AssignedAreas assignedAreas;

        // clear out assignments from prior frame
        map<CSGOId, AreaId> oldDangerAreaIds = blackboard.playerToDangerAreaId;
        blackboard.playerToDangerAreaId.clear();
        AreaBits ctLastDangerAreas, tLastDangerAreas;
        for (const auto & [playerId, dangerAreaId] : oldDangerAreaIds) {
            const auto & client = state.getClient(playerId);
            if (client.isAlive && client.isBot) {
                if (client.team == ENGINE_TEAM_CT) {
                    ctLastDangerAreas.set(blackboard.navFile.m_area_ids_to_indices[dangerAreaId], true);
                }
                else if (client.team == ENGINE_TEAM_T) {
                    tLastDangerAreas.set(blackboard.navFile.m_area_ids_to_indices[dangerAreaId], true);
                }
            }
        }

        AreaBits tVisibleAreas = blackboard.getVisibleAreasByTeam(state, ENGINE_TEAM_T),
                ctVisibleAreas = blackboard.getVisibleAreasByTeam(state, ENGINE_TEAM_CT);

        for (const auto & client : state.clients) {
            if (client.isAlive && client.isBot) {
                AreaBits & teamAssignedAreas = assignedAreas.getTeamAssignedAreas(client);
                vector<CSKnowTime> & dangerAreaLastCheckTime = blackboard.getDangerAreaLastCheckTime(client);

                const nav_mesh::nav_area & curArea =
                        blackboard.navFile.get_nearest_area_by_position(vec3Conv(client.getFootPosForPlayer()));

                // filter for cover edge - visible, but adjacent to behind cover area
                // can't use precomputed danger areas by area as want to account for visibility from multiple teammates
                vector<CoverEdge> coverEdges;
                AreaBits boolCoverEdges;
                for (size_t i = 0; i < blackboard.navFile.m_areas.size(); i++) {
                    // bot can't watch an area it can't personally see
                    if (!blackboard.visPoints.isVisibleAreaId(curArea.get_id(), blackboard.navFile.m_areas[i].get_id())) {
                        continue;
                    }
                    if ((client.team == ENGINE_TEAM_T && tVisibleAreas[i]) || (client.team == ENGINE_TEAM_CT && ctVisibleAreas[i])) {
                        for (size_t j = 0; j < blackboard.navFile.connections_area_length[i]; j++) {
                            size_t conAreaIndex = blackboard.navFile.connections[blackboard.navFile.connections_area_start[i] + j];
                            if ((client.team == ENGINE_TEAM_T && !tVisibleAreas[conAreaIndex]) || (client.team == ENGINE_TEAM_CT && !ctVisibleAreas[conAreaIndex])) {
                                // distance is distance to possible enemy locations
                                double sumDistance = 0., minDistance = std::numeric_limits<double>::max();
                                for (const auto & possibleAreaIndex : blackboard.possibleNavAreas.getEnemiesPossiblePositions(state, client.csgoId)) {
                                    double newDistance = blackboard.reachability.getDistance(possibleAreaIndex, i);
                                    sumDistance += newDistance;
                                    minDistance = std::min(minDistance, newDistance);
                                }
                                // if there are no enemies or none nearby, then no need to worry about cover edges
                                // ACTUALLY, allow all, will just sort by min distance so get best cover edge
                                //if (sumDistance != 0. && secondsAwayAtMaxSpeed(minDistance) < 1.0) {
                                    coverEdges.push_back({i, blackboard.navFile.m_areas[i].get_id(), sumDistance,
                                                          secondsAwayAtMaxSpeed(minDistance),
                                                          state.getSecondsBetweenTimes(dangerAreaLastCheckTime[i], state.loadTime) < RECENTLY_CHECKED_SECONDS});
                                    boolCoverEdges.set(i, true);
                                //}
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

                // sort cover edges by possibility of enemy being there, then if chcked recently, and finally distance to mass of enemies
                std::sort(coverEdges.begin(), coverEdges.end(),
                          [](const CoverEdge & a, const CoverEdge & b) {
                    return (a.minTimeToEnemy < b.minTimeToEnemy) ||
                        (a.minTimeToEnemy == b.minTimeToEnemy && !a.checkedRecently && b.checkedRecently) ||
                        (a.minTimeToEnemy == b.minTimeToEnemy && a.checkedRecently == b.checkedRecently && a.distance < b.distance);
                });

                // find first non-assigned cover edge
                // if all already assigned, take the closest one
                // only reassign every DANGER_ATTENTION_SECONDS to ensure consistency of attention (assuming still a valid danger area)
                const auto & lastDangerAreas = client.team == ENGINE_TEAM_CT ? ctLastDangerAreas : tLastDangerAreas;
                if (blackboard.lastDangerAssignment.find(client.csgoId) == blackboard.lastDangerAssignment.end() ||
                    !boolCoverEdges[blackboard.navFile.m_area_ids_to_indices[oldDangerAreaIds[client.csgoId]]] ||
                    state.getSecondsBetweenTimes(blackboard.lastDangerAssignment[client.csgoId], state.loadTime) > DANGER_ATTENTION_SECONDS) {
                    blackboard.lastDangerAssignment[client.csgoId] = state.loadTime;
                    for (const auto & coverEdge : coverEdges) {
                        if (!teamAssignedAreas[coverEdge.areaIndex] && !lastDangerAreas[coverEdge.areaIndex]) {
                            blackboard.playerToDangerAreaId[client.csgoId] = coverEdge.areaId;
                            break;
                        }
                    }
                    if (blackboard.playerToDangerAreaId.find(client.csgoId) == blackboard.playerToDangerAreaId.end()) {
                        blackboard.playerToDangerAreaId[client.csgoId] = coverEdges[0].areaId;
                    }
                }
                else {
                    blackboard.playerToDangerAreaId[client.csgoId] = oldDangerAreaIds[client.csgoId];
                }

                size_t srcAreaIndex = blackboard.navFile.m_area_ids_to_indices[blackboard.playerToDangerAreaId[client.csgoId]];

                // check if already looking at assigned area. If so, update looking at time
                AABB dangerAABB(areaToAABB(blackboard.navFile.m_areas[srcAreaIndex]));
                dangerAABB.max.z += EYE_HEIGHT * 2;
                Ray playerRay = getEyeCoordinatesForPlayerGivenEyeHeight(client.getEyePosForPlayer(), client.getCurrentViewAngles());
                bool updateCheckTime = intersectP(dangerAABB, playerRay);

                // count all areas within distance to assigned one and also visible to player as also assigned
                for (size_t dstAreaIndex = 0; dstAreaIndex < blackboard.navFile.m_areas.size(); dstAreaIndex++) {
                    if (blackboard.reachability.getDistance(srcAreaIndex, dstAreaIndex) < WATCHED_DISTANCE &&
                        blackboard.visPoints.isVisibleAreaId(curArea.get_id(), blackboard.navFile.m_areas[dstAreaIndex].get_id())) {
                        if (updateCheckTime) {
                            dangerAreaLastCheckTime[dstAreaIndex] = state.loadTime;
                        }
                        teamAssignedAreas.set(dstAreaIndex, true);
                    }
                }
            }
        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
