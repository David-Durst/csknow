//
// Created by durst on 6/9/22.
//
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"
#include "bots/behavior_tree/priority/spacing_helpers.h"

namespace follow::spacing {
    bool BaitConditionNode::valid(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        const Order & curOrder = blackboard.strategy.getOrderForPlayer(treeThinker.csgoId);

        // stop if T team, not entering
        if (curClient.team == ENGINE_TEAM_T) {
            return false;
        }

        // stop if assigned entry index 0, not relevant then as either pusher or bait
        if (blackboard.strategy.playerToEntryIndex[treeThinker.csgoId] == 0) {
            return false;
        }

        int numAhead = computeNumAhead(blackboard, state, curClient);
        // ready to execute if right number of people are in front
        bool readyToExecture = numAhead = blackboard.strategy.playerToEntryIndex[treeThinker.csgoId];
        // stop if too few people are ahead
        return numAhead < blackboard.strategy.playerToEntryIndex[treeThinker.csgoId];
    }
}