//
// Created by durst on 5/8/22.
//

#include "bots/behavior_tree/action_node.h"

namespace action {
    NodeState MovementTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        Path & curPath = blackboard.playerToPath[treeThinker.csgoId];

        if (curPath.pathCallSucceeded) {

        }
    }
}
