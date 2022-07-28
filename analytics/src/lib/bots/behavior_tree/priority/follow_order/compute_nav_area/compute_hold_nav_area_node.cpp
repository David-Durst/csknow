//
// Created by durst on 6/9/22.
//
#include "bots/behavior_tree/priority/follow_order_node.h"
#include "bots/behavior_tree/priority/priority_helpers.h"

namespace follow::compute_nav_area {
    NodeState ComputeHoldNavAreaNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        AreaId curAreaId = blackboard.navFile.get_nearest_area_by_position(
                vec3Conv(state.getClient(treeThinker.csgoId).getFootPosForPlayer())).get_id();
        const Order & curOrder = blackboard.strategy.getOrderForPlayer(treeThinker.csgoId);
        OrderId curOrderId = blackboard.strategy.getOrderIdForPlayer(treeThinker.csgoId);

        // default values are set to invalid where necessary, so this is fine
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];

        curPriority.priorityType = PriorityType::Order;
        curPriority.targetPlayer.playerId = INVALID_ID;
        curPriority.moveOptions = {true, false, false};
        curPriority.shootOptions = ShootOptions::DontShoot;


        // for all nav areas, find the closest one to the hide place where the chokepoint is visible (break tie breakers by areaid)
        // for each one
        if (blackboard.strategy.getDistance(curAreaId, curOrderId,
                                            curOrder.playerToHoldIndex.find(treeThinker.csgoId)->second,
                                            blackboard.navFile, blackboard.reachability,
                                            blackboard.distanceToPlaces));


        playerNodeState[treeThinker.csgoId] = NodeState::Success;

        return playerNodeState[treeThinker.csgoId];
    }
}
