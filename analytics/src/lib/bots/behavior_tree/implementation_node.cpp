//
// Created by durst on 5/8/22.
//

#include "bots/behavior_tree/implementation_node.h"

namespace implementation {
    NodeState PathingTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];

        // check if priority's nav area is same. If so, do nothing
        if (blackboard.playerToPath.find(treeThinker.csgoId) != blackboard.playerToPath.end()) {
            Path & curPath = blackboard.playerToPath[treeThinker.csgoId];



            if (curPriority.getTargetNavArea(state, blackboard.navFile).get_id() != blackboard.navFile.m_areas
        }
    }

    NodeState FireSelectionTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        // not executing shooting if no target
        if (blackboard.playerToTarget.find(treeThinker.csgoId) == blackboard.playerToTarget.end()) {
            nodeState = NodeState::Failure;
            return nodeState;
        }

        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        const ServerState::Client & targetClient =
                state.getClient(blackboard.playerToTarget[treeThinker.csgoId].targetPlayer);
        double distance = computeDistance(curClient.getFootPosForPlayer(), targetClient.getFootPosForPlayer());

        // if close enough to move and shoot, crouch
        bool shouldCrouch = distance <= treeThinker.engagementParams.standDistance;
        if (distance <= treeThinker.engagementParams.moveDistance) {
            curPriority.movementOptions = {true, false, true};
            curPriority.shootOptions = PriorityShootOptions::Spray;
        }
        else if (distance <= treeThinker.engagementParams.sprayDistance) {
            curPriority.movementOptions = {false, false, shouldCrouch};
            curPriority.shootOptions = PriorityShootOptions::Spray;
        }
        else if (distance <= treeThinker.engagementParams.burstDistance) {
            curPriority.movementOptions = {false, false, shouldCrouch};
            curPriority.shootOptions = PriorityShootOptions::Burst;
        }
        else {
            curPriority.movementOptions = {false, false, shouldCrouch};
            curPriority.shootOptions = PriorityShootOptions::Tap;
        }
        nodeState = NodeState::Success;
        return nodeState;
    }
}

