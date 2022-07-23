//
// Created by steam on 7/11/22.
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
                int entryIndex = 0;
                for (const CSGOId followerId : order.followers) {
                    if (blackboard.playerToTreeThinkers[followerId].aggressiveType == AggressiveType::Push) {
                        blackboard.playerToEntryIndex[followerId] = entryIndex++;
                    }
                    else {
                        baitersOnOrder.push_back(followerId);
                    }
                }
                for (const CSGOId followerId : baitersOnOrder) {
                    blackboard.playerToEntryIndex[followerId] = entryIndex++;
                }
            }
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
};
