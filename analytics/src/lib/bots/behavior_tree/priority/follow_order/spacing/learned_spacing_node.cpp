//
// Created by durst on 4/22/23.
//

#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/analysis/learned_models.h"

namespace follow::spacing {
    NodeState LearnedSpacingNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        if (!blackboard.inAnalysis && !blackboard.inTest && usePlaceAreaModelProbabilities) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }
        return playerNodeState[treeThinker.csgoId];
    }
}
