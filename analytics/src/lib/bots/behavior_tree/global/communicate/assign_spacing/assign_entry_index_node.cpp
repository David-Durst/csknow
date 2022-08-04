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
            for (const auto & orderId : blackboard.strategy.getOrderIds(true, false)) {
                // assign one of pushers to go first, then assign rest
                // after pushers, assign baiters
                vector<CSGOId> baitersOnOrder;
                int entryIndex = 0;
                for (const CSGOId followerId : blackboard.strategy.getOrderFollowers(orderId)) {
                    if (blackboard.playerToTreeThinkers[followerId].aggressiveType == AggressiveType::Push) {
                        blackboard.strategy.playerToEntryIndex[followerId] = entryIndex++;
                        if (!blackboard.defuserId) {
                            blackboard.defuserId = followerId;
                        }
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
        blackboard.executeIfAllFinishedSetup(state);
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
};
