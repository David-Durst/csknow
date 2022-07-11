//
// Created by steam on 7/11/22.
//

#include "bots/behavior_tree/global/communicate_node.h"

namespace communicate {
    NodeState CommunicateTeamMemory::exec(const ServerState & state, TreeThinker &treeThinker) {
        blackboard.tMemory.updatePositions(state, blackboard.navFile, blackboard.tMemorySeconds);
        blackboard.ctMemory.updatePositions(state, blackboard.navFile, blackboard.ctMemorySeconds);
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
};
