//
// Created by steam on 7/4/22.
//

#include "bots/behavior_tree/priority/memory_node.h"

NodeState memory::PerPlayerMemory::exec(const ServerState & state, TreeThinker &treeThinker) {
    blackboard.playerToMemory[treeThinker.csgoId].updatePositions(state, blackboard.navFile, treeThinker.maxMemorySeconds);
    playerNodeState[treeThinker.csgoId] = NodeState::Running;
    return playerNodeState[treeThinker.csgoId];
}


