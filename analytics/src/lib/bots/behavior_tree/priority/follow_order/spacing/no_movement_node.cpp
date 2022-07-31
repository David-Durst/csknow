//
// Created by steam on 7/31/22.
//

#include "bots/behavior_tree/priority/follow_order_node.h"

namespace follow::spacing {
    NodeState NoMovementNode::exec(const ServerState & state, TreeThinker &treeThinker) {
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        curPriority.moveOptions.move = false;
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
