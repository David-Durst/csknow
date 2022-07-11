//
// Created by steam on 7/5/22.
//

#include "bots/behavior_tree/global/communicate_node.h"

namespace communicate {
    /**
     * Each order, assign players to push indices
     */
    NodeState AssignAggressionNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        if (blackboard.newOrderThisFrame) {
            for (const auto & order : blackboard.orders) {
                // assign one of pushers to go first, then assign rest
                // after pushers, assign baiters
                vector<CSGOId> baitersOnOrder;
                int pushIndex = 0;
                for (const CSGOId followerId : order.followers) {
                    if (blackboard.playerToTreeThinkers[followerId].aggressiveType == AggressiveType::Push) {
                        blackboard.playerToPushOrder[followerId] = pushIndex++;
                    }
                    else {
                        baitersOnOrder.push_back(followerId);
                    }
                }
                for (const CSGOId followerId : baitersOnOrder) {
                    blackboard.playerToPushOrder[followerId] = pushIndex++;
                }
            }
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

    NodeState CommunicateTeamMemory::exec(const ServerState & state, TreeThinker &treeThinker) {
        blackboard.tMemory.updatePositions(state, blackboard.navFile, blackboard.tMemorySeconds);
        blackboard.ctMemory.updatePositions(state, blackboard.navFile, blackboard.ctMemorySeconds);
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

    NodeState DiffusePositionsNode::exec(const ServerState & state, TreeThinker &treeThinker) {
        if (state.roundNumber != diffuseRoundNumber) {
            diffuseRoundNumber = state.roundNumber;
            for (const auto & client : state.clients) {
                blackboard.possibleNavAreas[client.csgoId].clear();
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

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }

};
