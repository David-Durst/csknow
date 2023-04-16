//
// Created by durst on 4/16/23.
//

#include "bots/behavior_tree/priority/follow_order_node.h"

namespace follow {
    NodeState EnemiesAliveTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        //const Order & curOrder = blackboard.strategy.getOrderForPlayer(treeThinker.csgoId);

        bool playerAliveOnDifferentTeam = false;
        for (const auto & client : state.clients) {
            if (client.isAlive && client.team != curClient.team) {
                playerAliveOnDifferentTeam = true;
            }
        }

        playerNodeState[treeThinker.csgoId] = playerAliveOnDifferentTeam ? NodeState::Success : NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
    }
}
