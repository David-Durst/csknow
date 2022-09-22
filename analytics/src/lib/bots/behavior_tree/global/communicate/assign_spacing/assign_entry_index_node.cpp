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

        if (!blackboard.defuserId) {
            for (const CSGOId playerId : state.getPlayersOnTeam(ENGINE_TEAM_CT)) {
                const ServerState::Client & client = state.getClient(playerId);
                if (client.isAlive && client.isBot) {
                    const nav_mesh::nav_area & playerNavArea = blackboard.getPlayerNavArea(client);
                    const nav_mesh::nav_area & c4NavArea = blackboard.getC4NavArea(state);
                    if (c4NavArea.m_place == playerNavArea.m_place) {
                        blackboard.defuserId = playerId;
                        break;
                    }
                }
            }
        }
        blackboard.executeIfAllFinishedSetup(state);
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
