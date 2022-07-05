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

};
