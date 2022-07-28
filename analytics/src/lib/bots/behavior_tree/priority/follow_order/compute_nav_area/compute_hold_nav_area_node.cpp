//
// Created by durst on 6/9/22.
//
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"

namespace follow::compute_nav_area {
    NodeState ComputeHoldNavAreaNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const Order & curOrder = blackboard.strategy.getOrderForPlayer(treeThinker.csgoId);

        // default values are set to invalid where necessary, so this is fine
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];

        curPriority.priorityType = PriorityType::Order;
        curPriority.targetPlayer.playerId = INVALID_ID;
        curPriority.moveOptions = {true, false, false};
        curPriority.shootOptions = ShootOptions::DontShoot;
        
        if ()


        playerNodeState[treeThinker.csgoId] = NodeState::Success;

        return playerNodeState[treeThinker.csgoId];
    }
}
