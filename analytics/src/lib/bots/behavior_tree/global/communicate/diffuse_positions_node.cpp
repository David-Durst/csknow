//
// Created by steam on 7/11/22.
//

#include "bots/behavior_tree/global/communicate_node.h"

namespace communicate {
    bitset<MAX_NAV_AREAS> getSpawnAreas(const ServerState & state, Blackboard & blackboard, const vector<CSGOId> & playersOnTeam) {
        CSKnowTime curTime = state.loadTime;
        bitset<MAX_NAV_AREAS> result;
        vector<CSGOId> alivePlayersOnTeam;
        for (const auto & id : playersOnTeam) {
            if (state.getClient(id).isAlive) {
                alivePlayersOnTeam.push_back(id);
            }
        }

        if (!alivePlayersOnTeam.empty()) {
            nav_mesh::vec3_t origPos = vec3Conv(state.getClient(alivePlayersOnTeam[0]).getFootPosForPlayer());
            AreaId origId = blackboard.navFile.get_nearest_area_by_position(origPos).get_id();
            result[blackboard.navFile.m_area_ids_to_indices[origId]] = true;
            for (size_t i = 1; i < alivePlayersOnTeam.size(); i++) {
                auto optionalWaypoints =
                        blackboard.navFile.find_path_detailed(origPos, vec3Conv(state.getClient(alivePlayersOnTeam[i]).getFootPosForPlayer()));
                if (optionalWaypoints) {
                    for (const auto & waypoint : optionalWaypoints.value()) {
                        result[blackboard.navFile.m_area_ids_to_indices[waypoint.area1]] = true;
                        if (waypoint.edgeMidpoint) {
                            result[blackboard.navFile.m_area_ids_to_indices[waypoint.area2]] = true;
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
            bitset<MAX_NAV_AREAS> tSpawnAreas = getSpawnAreas(state, blackboard, tPlayers);
            const vector<CSGOId> & ctPlayers = state.getPlayersOnTeam(ENGINE_TEAM_CT);
            bitset<MAX_NAV_AREAS> ctSpawnAreas = getSpawnAreas(state, blackboard, ctPlayers);

            // initialize nav areas to each as shortest path between all teammates, approximation of engine spawn zones
            for (const auto & client : state.clients) {
                if (client.team == ENGINE_TEAM_T) {
                    blackboard.possibleNavAreas.set(client.csgoId, tSpawnAreas, state.loadTime);
                }
                else if (client.team == ENGINE_TEAM_CT) {
                    blackboard.possibleNavAreas.set(client.csgoId, ctSpawnAreas, state.loadTime);
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
                blackboard.possibleNavAreas.reset(visibleEnemy.get().csgoId);
                AreaId curArea = blackboard.navFile.get_nearest_area_by_position(
                        vec3Conv(visibleEnemy.get().getFootPosForPlayer())).get_id();
                blackboard.possibleNavAreas.set(visibleEnemy.get().csgoId, curArea, true, curTime);
            }
        }

        // for each client, for each current area they could be, add all possible areas that could've been reached
        // since entering that area
        for (const auto & client : state.clients) {
            if (!client.isAlive) {
                blackboard.possibleNavAreas.reset(client.csgoId);
            }
            else {
                blackboard.possibleNavAreas.addNeighbors(state, blackboard.reachability, client.csgoId);
            }
        }

        /*
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
         */

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

};
