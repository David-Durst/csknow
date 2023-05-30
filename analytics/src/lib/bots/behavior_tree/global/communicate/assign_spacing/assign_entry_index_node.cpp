//
// Created by steam on 7/11/22.
//
#include "bots/behavior_tree/global/communicate_node.h"

namespace communicate::spacing {
    /**
     * Each order, assign players to push indices
     */
    NodeState AssignEntryIndexNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        if (blackboard.newOrderThisFrame) {
            blackboard.defuserId = {};
        }

        if (blackboard.newOrderThisFrame && !blackboard.modelOrdersCT) {
            for (const auto & orderId : blackboard.strategy.getOrderIds(false, true)) {
                // assign one of pushers to go first, then assign rest
                // after pushers, assign baiters
                vector<CSGOId> baitersOnOrder;
                int entryIndex = 0;
                for (const CSGOId followerId : blackboard.strategy.getOrderFollowers(orderId)) {
                    if (blackboard.playerToTreeThinkers[followerId].aggressiveType == AggressiveType::Push) {
                        blackboard.strategy.playerToEntryIndex[followerId] = entryIndex++;
                    }
                    else {
                        baitersOnOrder.push_back(followerId);
                    }
                }
                for (const CSGOId followerId : baitersOnOrder) {
                    blackboard.strategy.playerToEntryIndex[followerId] = entryIndex++;
                }
            }
        }


        // if no T alive, assign closest to C4
        bool tAlive = false, ctAlive = false;
        for (const auto & client : state.clients) {
            if (client.team == ENGINE_TEAM_T && client.isAlive) {
                tAlive = true;
                break;
            }
        }
        for (const auto & client : state.clients) {
            if (client.team == ENGINE_TEAM_CT && client.isAlive) {
                ctAlive = true;
                break;
            }
        }
        if (!tAlive && ctAlive) {
            double minDistance = std::numeric_limits<double>::max();
            int64_t closestClientId = INVALID_ID;
            for (const auto & client : state.clients) {
                if (client.team == ENGINE_TEAM_CT && client.isAlive) {
                    const nav_mesh::nav_area & playerNavArea = blackboard.getPlayerNavArea(client);
                    const nav_mesh::nav_area & c4NavArea = blackboard.getC4NavArea(state);
                    double distanceToC4 = blackboard.reachability.getDistance(playerNavArea.get_id(),
                                                                              c4NavArea.get_id(), blackboard.navFile);
                    if (distanceToC4 < minDistance) {
                        minDistance = distanceToC4;
                        closestClientId = client.csgoId;
                    }
                }
                blackboard.defuserId = closestClientId;
            }
        }
        blackboard.executeIfAllFinishedSetup(state);
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
