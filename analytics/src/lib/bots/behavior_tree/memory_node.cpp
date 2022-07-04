//
// Created by steam on 7/4/22.
//

#include "bots/behavior_tree/memory_node.h"

NodeState memory::PerPlayerMemory::exec(const ServerState & state, TreeThinker &treeThinker) {
    blackboard.playerToMemory[treeThinker.csgoId].updatePositions(state, blackboard.navFile, treeThinker.maxMemorySeconds);
    playerNodeState[treeThinker.csgoId] = NodeState::Success;
    return playerNodeState[treeThinker.csgoId];
}

NodeState memory::CommunicateTeamMemory::exec(const ServerState & state, TreeThinker &treeThinker) {
    if (blackboard.lastCommunicateFrame != state.getClient(treeThinker.csgoId).lastFrame) {
        blackboard.tMemory.updatePositions(state, blackboard.navFile, blackboard.tMemorySeconds);
        blackboard.ctMemory.updatePositions(state, blackboard.navFile, blackboard.ctMemorySeconds);
    }
    playerNodeState[treeThinker.csgoId] = NodeState::Success;
    return playerNodeState[treeThinker.csgoId];
}

