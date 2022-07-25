//
// Created by durst on 6/9/22.
//
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"

namespace follow {
    NodeState ComputeObjectiveAreaNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const Order & curOrder = blackboard.strategy.getOrderForPlayer(treeThinker.csgoId);
        bool havePriority = blackboard.playerToPriority.find(treeThinker.csgoId) != blackboard.playerToPriority.end();

        // default values are set to invalid where necessary, so this is fine
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];

        // if no priority yet or switching from engagement, setup priority without a target
        //if (!havePriority || curPriority.priorityType != PriorityType::Order) {
            moveToWaypoint(blackboard, state, treeThinker, curOrder, curPriority);
            curPriority.priorityType = PriorityType::Order;
            curPriority.targetPlayer.playerId = INVALID_ID;
            curPriority.moveOptions = {true, false, false};
            curPriority.shootOptions = ShootOptions::DontShoot;
        //}

        string curPlace = blackboard.getPlayerPlace(state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer());
        int64_t maxFinishedWaypoint = getMaxFinishedWaypoint(state, curOrder, curPriority, curPlace);

        bool finishedWaypoint = false;
        // if finished waypoint,
        if (maxFinishedWaypoint >= blackboard.strategy.playerToWaypointIndex[treeThinker.csgoId]) {
            finishedWaypoint = true;
            // need to pick a new path on priority change
            blackboard.playerToPath.erase(treeThinker.csgoId);
            // increment counter and move to next waypoint if possible
            if (maxFinishedWaypoint < curOrder.waypoints.size() - 1) {
                blackboard.strategy.playerToWaypointIndex[treeThinker.csgoId] = maxFinishedWaypoint + 1;
                moveToWaypoint(blackboard, state, treeThinker, curOrder, curPriority);
            }
        }


        const Action &priorAction = blackboard.lastPlayerToAction[treeThinker.csgoId];
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);

        // fail pushing if stuck
        if (havePriority && !finishedWaypoint && priorAction.moving() && computeMagnitude(curClient.getVelocity()) < MOVING_THRESHOLD) {
            //curPriority.stuckTicks++;
            curPriority.stuckTicks = 0;
        }
        else {
            curPriority.stuckTicks = 0;
        }

        if (curPriority.stuckTicks > STUCK_TICKS_THRESHOLD) {
            curPriority.stuckTicks = 0;
            curPriority.stuck = true;
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
        }

        return playerNodeState[treeThinker.csgoId];
    }
}
