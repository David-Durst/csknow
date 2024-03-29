//
// Created by steam on 7/11/22.
//

#include "bots/behavior_tree/global/communicate_node.h"

namespace communicate {
    AreaBits getSpawnAreas(const ServerState & state, Blackboard & blackboard, const vector<CSGOId> & playersOnTeam) {
        AreaBits result;
        vector<CSGOId> alivePlayersOnTeam;
        for (const auto & id : playersOnTeam) {
            if (state.getClient(id).isAlive) {
                alivePlayersOnTeam.push_back(id);
            }
        }

        if (!alivePlayersOnTeam.empty()) {
            nav_mesh::vec3_t origPos = vec3Conv(state.getClient(alivePlayersOnTeam[0]).getFootPosForPlayer());
            AreaId origId = blackboard.navFile.get_nearest_area_by_position(origPos).get_id();
            result.set(blackboard.navFile.m_area_ids_to_indices[origId], true);
            for (size_t i = 1; i < alivePlayersOnTeam.size(); i++) {
                auto optionalWaypoints =
                        blackboard.navFile.find_path_detailed(origPos, vec3Conv(state.getClient(alivePlayersOnTeam[i]).getFootPosForPlayer()));
                if (optionalWaypoints) {
                    for (const auto & waypoint : optionalWaypoints.value()) {
                        result.set(blackboard.navFile.m_area_ids_to_indices[waypoint.area1], true);
                        if (waypoint.edgeMidpoint) {
                            result.set(blackboard.navFile.m_area_ids_to_indices[waypoint.area2], true);
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
            AreaBits tSpawnAreas = getSpawnAreas(state, blackboard, tPlayers);
            const vector<CSGOId> & ctPlayers = state.getPlayersOnTeam(ENGINE_TEAM_CT);
            AreaBits ctSpawnAreas = getSpawnAreas(state, blackboard, ctPlayers);

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
            if (tPlayers.size() + ctPlayers.size() == static_cast<size_t>(state.numPlayersAlive()) ||
                blackboard.resetPossibleNavAreas) {
                diffuseRoundNumber = state.roundNumber;
                blackboard.resetPossibleNavAreas = false;
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

        // for each client, for each area they could be, remove all areas that are visible to enemies
        // first get areas visible to enemies
        AreaBits tVisibleAreas = blackboard.getVisibleAreasByTeam(state, ENGINE_TEAM_T),
            ctVisibleAreas = blackboard.getVisibleAreasByTeam(state, ENGINE_TEAM_CT);

        // also get areas reachable from cur area in one step
        // if reachable in one step, at least part of it must be visible
        for (const auto & client : state.clients) {
            if (client.isAlive) {
                AreaId areaId = blackboard.getPlayerNavArea(client).get_id();
                size_t areaIndex = blackboard.navFile.m_area_ids_to_indices[areaId];
                for (size_t i = 0; i < blackboard.navFile.connections_area_length[areaIndex]; i++) {
                    size_t conAreaIndex = blackboard.navFile.connections[
                            blackboard.navFile.connections_area_start[areaIndex] + i];
                    if (client.team == ENGINE_TEAM_T) {
                        tVisibleAreas.set(conAreaIndex, true);
                    }
                    else if (client.team == ENGINE_TEAM_CT) {
                        ctVisibleAreas.set(conAreaIndex, true);
                    }
                }
            }
        }

        // flip visible areas to got not visible areas
        tVisibleAreas.flip();
        ctVisibleAreas.flip();

        // for each player, and to get rid of visible areas
        for (const auto & client : state.clients) {
            if (client.isAlive) {
                if (client.team == ENGINE_TEAM_T) {
                    blackboard.possibleNavAreas.andBits(client.csgoId, ctVisibleAreas);
                }
                if (client.team == ENGINE_TEAM_CT) {
                    blackboard.possibleNavAreas.andBits(client.csgoId, tVisibleAreas);
                }
            }
        }

        // fix positions of players visible to enemies
        CSKnowTime curTime = state.loadTime;
        for (const auto & client : state.clients) {
            const auto visibleEnemies = state.getVisibleEnemies(client.csgoId);
            for (const auto & visibleEnemy : visibleEnemies) {
                blackboard.possibleNavAreas.reset(visibleEnemy.get().csgoId);
                AreaId curArea = blackboard.navFile.get_nearest_area_by_position(
                        vec3Conv(visibleEnemy.get().getFootPosForPlayer())).get_id();
                blackboard.possibleNavAreas.set(visibleEnemy.get().csgoId, curArea, true, curTime);
            }
        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

}
