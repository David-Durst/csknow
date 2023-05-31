//
// Created by durst on 6/9/22.
//
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"
#include "bots/behavior_tree/priority/spacing_helpers.h"

namespace follow::spacing {
    bool BaitConditionNode::valid(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        //const Order & curOrder = blackboard.strategy.getOrderForPlayer(treeThinker.csgoId);

        // stop if T team, not entering
        if (curClient.team == ENGINE_TEAM_T) {
            return false;
        }

        // stop if assigned entry index 0, not relevant then as either pusher or lurk
        if (blackboard.strategy.playerToEntryIndex[treeThinker.csgoId] == 0) {
            return false;
        }

        NumAheadResult numAheadResult = computeNumAhead(blackboard, state, curClient);
        // ready to execute if enough people ahead and near enough to next person in stack
        bool allPlayersInFrontFinished = true;
        int32_t curPlayerEntryIndex = blackboard.strategy.playerToEntryIndex[treeThinker.csgoId];
        const OrderId & curOrderId = blackboard.strategy.getOrderIdForPlayer(curClient.csgoId);
        for (const auto & csgoId : blackboard.strategy.getOrderFollowers(curOrderId)) {
            if (blackboard.strategy.playerToEntryIndex[csgoId] < curPlayerEntryIndex &&
                !blackboard.strategy.playerFinishedSetup(csgoId)) {
                allPlayersInFrontFinished = false;
            }
        }
        bool readyToExecute = numAheadResult.numAhead == blackboard.strategy.playerToEntryIndex[treeThinker.csgoId] &&
                (numAheadResult.nearestInFront < MAX_BAIT_DISTANCE || allPlayersInFrontFinished);
        if (readyToExecute) {
            blackboard.strategy.playerReady(treeThinker.csgoId);
        }
        else {
            // this will ignore execute -> setup transition, so fine to call many times
            blackboard.strategy.playerSetup(treeThinker.csgoId);
        }
        return numAheadResult.numAhead < blackboard.strategy.playerToEntryIndex[treeThinker.csgoId] ||
                (numAheadResult.numAhead == blackboard.strategy.playerToEntryIndex[treeThinker.csgoId] &&
                numAheadResult.nearestInFront < MIN_BAIT_DISTANCE);
    }
}