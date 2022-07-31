//
// Created by steam on 7/31/22.
//

#include "bots/behavior_tree/priority/engage_node.h"

namespace engage {
    NodeState RecordEngagementRound::exec(const ServerState &state, TreeThinker &treeThinker) {
        blackboard.teamToLastRoundSawEnemy[state.getClient(treeThinker.csgoId).team] = state.roundNumber;
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
