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

        // default values are set to invalid where necessary, so this is fine
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        curPriority.targetAreaId = curOrder.holdIndexToHoldAreaId.find(curOrder.playerToHoldIndex.find(treeThinker.csgoId)->second)->second;
        curPriority.targetPos = vec3tConv(blackboard.navFile.get_area_by_id_fast(curPriority.targetAreaId).get_center());
        curPriority.priorityType = PriorityType::Order;
        curPriority.targetPlayer.playerId = INVALID_ID;
        curPriority.moveOptions = {true, false, false};
        curPriority.shootOptions = ShootOptions::DontShoot;

        if (blackboard.inTest) {
            int x = 1;
        }

        // if in the target area, don't move
        if (curAreaId == curPriority.targetAreaId) {
            curPriority.moveOptions = {false, false, false};
        }

        // aim at danger area if visible
        AreaId dangerAreaId = curOrder.holdIndexToDangerAreaId.find(
            curOrder.playerToHoldIndex.find(treeThinker.csgoId)->second)->second;
        if (blackboard.visPoints.isVisibleAreaId(curAreaId, dangerAreaId)) {
            blackboard.playerToDangerAreaId[treeThinker.csgoId] = dangerAreaId;
        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
