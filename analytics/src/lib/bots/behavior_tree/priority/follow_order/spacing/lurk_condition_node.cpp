//
// Created by durst on 6/9/22.
//
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"
#include "bots/behavior_tree/priority/spacing_helpers.h"

namespace follow::spacing {
    bool LurkConditionNode::valid(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        OrderId curOrderId = blackboard.strategy.getOrderIdForPlayer(treeThinker.csgoId);
        //const Order & curOrder = blackboard.strategy.getOrder(curOrderId);
        const vector<CSGOId> & curOrderFollowers = blackboard.strategy.getOrderFollowers(curOrderId);

        // stop if T team, not entering
        if (curClient.team == ENGINE_TEAM_T) {
            return false;
        }

        // entry index must be 0 and be a baiter to lurk, otherwise baiting or pushing
        if (blackboard.strategy.playerToEntryIndex[treeThinker.csgoId] != 0 ||
            treeThinker.aggressiveType != AggressiveType::Bait) {
            return false;
        }

        NumAheadResult numAheadResult = computeNumAhead(blackboard, state, curClient);
        // ready if reached first waypoint (aka on to second/number 1 by 0 indexing) and in front
        bool readyToExecute = numAheadResult.numBehind == static_cast<int>(curOrderFollowers.size()) - 1 &&
                              numAheadResult.nearestBehind > MIN_BAIT_DISTANCE &&
                              blackboard.strategy.playerToWaypointIndex[treeThinker.csgoId] > 0;
        if (readyToExecute) {
            blackboard.strategy.playerReady(treeThinker.csgoId);
        }
        else {
            // this will ignore execute -> setup transition, so fine to call many times
            blackboard.strategy.playerSetup(treeThinker.csgoId);
        }
        // stop only when ready and too far ahead (so this player is not the problem, need to wait for others to catch up)
        // or pusher hasn't seen anyone
        //bool sawEnemyThisRound = blackboard.sawEnemyThisRound(state, curClient.team);
        /*
        if (blackboard.strategy.isPlayerExecuting(treeThinker.csgoId)) {
            int x= 1;
        }
         */
        return blackboard.strategy.isPlayerReady(treeThinker.csgoId) ||
                (blackboard.strategy.isPlayerExecuting(treeThinker.csgoId) && (
                        (curOrderFollowers.size() > 1 && numAheadResult.nearestBehind > MAX_PUSH_DISTANCE) ||
                        !blackboard.sawEnemyThisRound(state, curClient.team)));
    }
}