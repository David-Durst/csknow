//
// Created by steam on 7/11/22.
//

#include "bots/behavior_tree/global/communicate_node.h"

namespace communicate {
    map<AreaId, CSKnowTime> getSpawnAreas(const ServerState & state, Blackboard & blackboard, const vector<CSGOId> & playersOnTeam) {
        CSKnowTime curTime = state.loadTime;
        map<AreaId, CSKnowTime> result;
        vector<CSGOId> alivePlayersOnTeam;
        for (const auto & id : playersOnTeam) {
            if (state.getClient(id).isAlive) {
                alivePlayersOnTeam.push_back(id);
            }
        }

        if (!alivePlayersOnTeam.empty()) {
            nav_mesh::vec3_t origPos = vec3Conv(state.getClient(alivePlayersOnTeam[0]).getFootPosForPlayer());
            result[blackboard.navFile.get_nearest_area_by_position(origPos).get_id()] = curTime;
            for (size_t i = 1; i < alivePlayersOnTeam.size(); i++) {
                auto optionalWaypoints =
                        blackboard.navFile.find_path_detailed(origPos, vec3Conv(state.getClient(alivePlayersOnTeam[i]).getFootPosForPlayer()));
                if (optionalWaypoints) {
                    for (const auto & waypoint : optionalWaypoints.value()) {
                        result[waypoint.area1] = curTime;
                        if (waypoint.edgeMidpoint) {
                            result[waypoint.area2] = curTime;
                        }
                    }
                }
            }
        }
        return result;
    }

    NodeState DiffusePositionsNode::exec(const ServerState & state, TreeThinker &treeThinker) {
        if (state.roundNumber != diffuseRoundNumber || blackboard.resetPossibleNavAreas) {
            const vector<CSGOId> & tPlayers = state.getPlayersOnTeam(ENGINE_TEAM_T);
            map<AreaId, CSKnowTime> tSpawnAreas = getSpawnAreas(state, blackboard, tPlayers);
            const vector<CSGOId> & ctPlayers = state.getPlayersOnTeam(ENGINE_TEAM_CT);
            map<AreaId, CSKnowTime> ctSpawnAreas = getSpawnAreas(state, blackboard, ctPlayers);

            // initialize nav areas to each as shortest path between all teammates, approximation of engine spawn zones
            for (const auto & client : state.clients) {
                if (client.team == ENGINE_TEAM_T) {
                    blackboard.possibleNavAreas[client.csgoId] = tSpawnAreas;
                }
                else if (client.team == ENGINE_TEAM_CT) {
                    blackboard.possibleNavAreas[client.csgoId] = ctSpawnAreas;
                }
            }

            // rerun until get first tick of round where everyone is alive
            if (tPlayers.size() + ctPlayers.size() == state.numPlayersAlive() || blackboard.resetPossibleNavAreas) {
                diffuseRoundNumber = state.roundNumber;
                blackboard.resetPossibleNavAreas = false;
            }
        }

        CSKnowTime curTime = state.loadTime;
        set<CSKnowId> visibleToEnemies;
        // fix positions of players visible to enemies
        for (const auto & client : state.clients) {
            const auto visibleEnemies = state.getVisibleEnemies(client.csgoId);
            for (const auto & visibleEnemy : visibleEnemies) {
                visibleToEnemies.insert(visibleEnemy.get().csgoId);
                blackboard.possibleNavAreas[visibleEnemy.get().csgoId].clear();
                AreaId curArea = blackboard.navFile.get_nearest_area_by_position(
                        vec3Conv(visibleEnemy.get().getFootPosForPlayer())).get_id();
                blackboard.possibleNavAreas[visibleEnemy.get().csgoId][curArea] = curTime;
            }
        }

        // for each client, for each current area they could be, add all possible areas that could've been reached
        // since entering that area
        for (const auto & client : state.clients) {
            if (!client.isAlive) {
                blackboard.possibleNavAreas[client.csgoId].clear();
            }
            else {
                auto & playerPossibleNavAreas = blackboard.possibleNavAreas[client.csgoId];
                set<AreaId> connectionsToAdd;
                for (const auto & [possibleAreaId, entryTime] : playerPossibleNavAreas) {
                    Vec3 curCenter = vec3tConv(blackboard.navFile.get_area_by_id_fast(possibleAreaId).get_center());
                    for (const auto & connection : blackboard.navFile.get_area_by_id_fast(possibleAreaId).get_connections()) {
                        if (playerPossibleNavAreas.find(connection.id) == playerPossibleNavAreas.end()) {
                            double connectionDistance = computeDistance(curCenter,
                                                                        vec3tConv(blackboard.navFile.get_area_by_id_fast(connection.id).get_center()));
                            if (connectionDistance / MAX_RUN_SPEED <= state.getSecondsBetweenTimes(entryTime, curTime)) {
                                connectionsToAdd.insert(connection.id);
                            }
                        }
                    }
                }

                for (const auto & connectionToAdd : connectionsToAdd) {
                    playerPossibleNavAreas[connectionToAdd] = curTime;
                }
            }
        }

        // for each client, for each area they could be, remove all areas that are visible to enemies
        // first build set of visible points for each team
        set<AreaId> tCurAreas, ctCurAreas;
        for (const auto & client : state.clients) {
            if (client.isAlive) {
                AreaId curArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(client.getFootPosForPlayer())).get_id();
                if (client.team == ENGINE_TEAM_T) {
                    tCurAreas.insert(curArea);
                }
                else if (client.team == ENGINE_TEAM_CT) {
                    ctCurAreas.insert(curArea);
                }
            }
        }
        set<AreaId> tVisibleAreas = blackboard.visPoints.getAreasRelativeToSrc(tCurAreas, true),
            ctVisibleAreas = blackboard.visPoints.getAreasRelativeToSrc(ctCurAreas, true);
        // for each player, build list of possible areas that intersect with a visible area, then remove that list
        map<CSGOId, vector<AreaId>> areasToRemove;
        for (const auto & client : state.clients) {
            if (client.isAlive) {
                for (const auto & [areaId, _] : blackboard.possibleNavAreas[client.csgoId]) {
                    if (blackboard.navFile.get_nearest_area_by_position(vec3Conv(client.getFootPosForPlayer())).get_id() ==
                        areaId) {
                        continue;
                    }
                    if (client.team == ENGINE_TEAM_T && ctVisibleAreas.find(areaId) != ctVisibleAreas.end()) {
                        areasToRemove[client.csgoId].push_back(areaId);
                    }
                    if (client.team == ENGINE_TEAM_CT && tVisibleAreas.find(areaId) != tVisibleAreas.end()) {
                        areasToRemove[client.csgoId].push_back(areaId);
                    }
                }
            }
        }
        for (const auto & [csgoId, playerAreasToRemove] : areasToRemove) {
            for (const auto & playerAreaToRemove : playerAreasToRemove) {
                blackboard.possibleNavAreas[csgoId].erase(playerAreaToRemove);
            }
        }
        if (blackboard.inTest) {
            for (const auto & client : state.clients) {
                if (blackboard.possibleNavAreas[client.csgoId].find(4182) != blackboard.possibleNavAreas[client.csgoId].end() && client.team == ENGINE_TEAM_T) {
                    bool rightArea = tVisibleAreas.find(4182) != ctVisibleAreas.end();
                    std::cout << rightArea << std::endl;
                }
            }
        }


        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

};
