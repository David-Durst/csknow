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
        /*
         * note: on enemy death, will get blackboard cleared, so only way to tell is new order rather than
         * if old priorty was engagement
        if (blackboard.newOrderThisFrame) {
            std::cout << "switching this frame" << std::endl;
        }
         */
        curPriority.priorityType = PriorityType::Order;
        curPriority.targetPlayer.playerId = INVALID_ID;
        curPriority.nonDangerAimArea = {};
        curPriority.moveOptions = {true, false, false};
        curPriority.shootOptions = ShootOptions::DontShoot;

        if (blackboard.strategy.playersFinishedStrategy.count(treeThinker.csgoId) > 0) {
            blackboard.strategy.playersFinishedStrategy.erase(treeThinker.csgoId);
        }

        /*
        const auto & curArea = blackboard.navFile.get_nearest_area_by_position(
            vec3Conv(state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer()));
        AreaId curAreaId = curArea.get_id();
        std::cout << "curAreaId: " << curAreaId << ", curArea.m_id " << curArea.m_id << std::endl;
         */
        AreaId curAreaId = blackboard.navFile.get_nearest_area_by_position(
                vec3Conv(state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer())).get_id();
        string curPlace = blackboard.getPlayerPlace(state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]].getFootPosForPlayer());
        int64_t maxFinishedWaypoint = getMaxFinishedWaypoint(blackboard, state, curOrder, treeThinker.csgoId, curPlace, curAreaId);

        // if finished waypoint,
        if (maxFinishedWaypoint >= blackboard.strategy.playerToWaypointIndex[treeThinker.csgoId]) {
            // need to pick a new path on priority change
            blackboard.playerToPath.erase(treeThinker.csgoId);
            // increment counter and move to next waypoint if possible
            if (maxFinishedWaypoint < static_cast<int64_t>(curOrder.waypoints.size()) - 1) {
                blackboard.strategy.playerToWaypointIndex[treeThinker.csgoId] = maxFinishedWaypoint + 1;
                moveToWaypoint(blackboard, state, treeThinker, curOrder, curPriority);
            }
            // otherwise, stop (if in air, could be landing, so keep going forward then)
            else {
                blackboard.strategy.playersFinishedStrategy.insert(treeThinker.csgoId);
                // if interrupted while defusing, when swithc back make sure to return to final waypont and not initial entry
                blackboard.strategy.playerToWaypointIndex[treeThinker.csgoId] =
                    static_cast<int64_t>(curOrder.waypoints.size()) - 1;
                if (!state.getClient(treeThinker.csgoId).isAirborne) {
                    curPriority.moveOptions.move = false;
                }
            }
        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
