//
// Created by durst on 6/9/22.
//
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"
#include "bots/behavior_tree/priority/spacing_helpers.h"

namespace follow::spacing {
    bool PushConditionNode::valid(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        const Order & curOrder = blackboard.strategy.getOrderForPlayer(treeThinker.csgoId);

        // stop if T team, not entering
        if (curClient.team == ENGINE_TEAM_T) {
            return false;
        }

        // entry index must be 0 and be a pusher to really be pusher, otherwise temporary baiting or baiting or lurking
        if (blackboard.strategy.playerToEntryIndex[treeThinker.csgoId] != 0 ||
            treeThinker.aggressiveType != AggressiveType::Push) {
            return false;
        }

        NumAheadResult numAheadResult = computeNumAhead(blackboard, state, curClient);
        // stop if reached first waypoint and first
        bool readyToExecute = numAheadResult.numAhead == 0 &&
                blackboard.strategy.playerToWaypointIndex[treeThinker.csgoId] > 0;
        if (readyToExecute) {
            blackboard.strategy.playerReady(treeThinker.csgoId);
        }
        else {
            // this will ignore execute -> setup transition, so fine to call many times
            blackboard.strategy.playerSetup(treeThinker.csgoId);
        }
        // when executing, just go, or when setting up
        return blackboard.strategy.playerNotReady(treeThinker.csgoId);
    }
}