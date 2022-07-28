//
// Created by durst on 6/9/22.
//
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"

namespace follow::compute_nav_area {
    NodeState ComputeEntryNavAreaNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const Order & curOrder = blackboard.strategy.getOrderForPlayer(treeThinker.csgoId);

        // default values are set to invalid where necessary, so this is fine
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];

        // if no priority yet or switching from engagement, setup priority without a target
        // just do this every frame since cheap, less conditions through tree to track
        moveToWaypoint(blackboard, state, treeThinker, curOrder, curPriority);
        curPriority.priorityType = PriorityType::Order;
        curPriority.targetPlayer.playerId = INVALID_ID;
        curPriority.moveOptions = {true, false, false};
        curPriority.shootOptions = ShootOptions::DontShoot;

        string curPlace = blackboard.getPlayerPlace(state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer());
        int64_t maxFinishedWaypoint = getMaxFinishedWaypoint(state, curOrder, curPriority, curPlace);

        // if finished waypoint,
        if (maxFinishedWaypoint >= blackboard.strategy.playerToWaypointIndex[treeThinker.csgoId]) {
            // need to pick a new path on priority change
            blackboard.playerToPath.erase(treeThinker.csgoId);
            // increment counter and move to next waypoint if possible
            if (maxFinishedWaypoint < curOrder.waypoints.size() - 1) {
                blackboard.strategy.playerToWaypointIndex[treeThinker.csgoId] = maxFinishedWaypoint + 1;
                moveToWaypoint(blackboard, state, treeThinker, curOrder, curPriority);
            }
            // otherwise, stop
            else {
                curPriority.moveOptions.move = false;
            }
        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;

        return playerNodeState[treeThinker.csgoId];
    }
}
