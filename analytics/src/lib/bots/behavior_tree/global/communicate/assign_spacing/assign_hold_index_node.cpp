//
// Created by steam on 7/11/22.
//
#include "bots/behavior_tree/global/communicate_node.h"

namespace communicate::spacing {
    struct HoldOption {
        size_t waypointIndex;
        bool aggressive;
        int numPlayers;
    };
    /**
     * Each order, assign players to hold indexes for waypoints in order
     */
    NodeState AssignHoldIndexNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        if (blackboard.newOrderThisFrame) {
            for (const auto & orderId : blackboard.strategy.getOrderIds(true, false)) {
                // assign to different hold points, prefering aggro for pushers and passive for baiters
                const Order & order = blackboard.strategy.getOrder(orderId);
                const Waypoints waypoints = order.waypoints;
                vector<HoldOption> holdOptions;
                for (size_t i = 0; i < order.holdIndices.size(); i++) {
                    holdOptions.push_back({order.holdIndices[i],
                                           order.waypoints[order.holdIndices[i]].aggresiveDefense, 0});
                }

                for (const CSGOId followerId : blackboard.strategy.getOrderFollowers(orderId)) {
                    if (blackboard.playerToTreeThinkers[followerId].aggressiveType == AggressiveType::Push) {
                        std::sort(holdOptions.begin(), holdOptions.end(),
                                  [](const HoldOption & a, const HoldOption & b)
                                  { return (a.numPlayers < b.numPlayers) ||
                                        (a.numPlayers == b.numPlayers && a.aggressive && !b.aggressive) ||
                                        (a.numPlayers == b.numPlayers && a.aggressive == b.aggressive && a.waypointIndex < b.waypointIndex); });
                    }
                    else {
                        std::sort(holdOptions.begin(), holdOptions.end(),
                                  [](const HoldOption & a, const HoldOption & b)
                                  { return (a.numPlayers < b.numPlayers) ||
                                           (a.numPlayers == b.numPlayers && !a.aggressive && b.aggressive) ||
                                           (a.numPlayers == b.numPlayers && a.aggressive == b.aggressive && a.waypointIndex < b.waypointIndex); });
                    }
                    blackboard.strategy.assignPlayerToHoldIndex(followerId, orderId, holdOptions.front().waypointIndex);
                    holdOptions.erase(holdOptions.begin());
                }
            }
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
};
