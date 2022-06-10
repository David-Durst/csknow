//
// Created by durst on 6/9/22.
//

#include "bots/behavior_tree/priority/engage_node.h"

namespace engage {
    NodeState SelectFireModeNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        Path & curPath = blackboard.playerToPath[treeThinker.csgoId];

        // not executing shooting if no target
        if (curPriority.targetPlayer.playerId == INVALID_ID) {
            curPath.movementOptions = {true, false, false};
            curPath.shootOptions = PathShootOptions::DontShoot;

            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }

        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        const ServerState::Client & targetClient = state.getClient(curPriority.targetPlayer.playerId);
        double distance = computeDistance(curClient.getFootPosForPlayer(), targetClient.getFootPosForPlayer());

        // if close enough to move and shoot, crouch
        bool shouldCrouch = distance <= treeThinker.engagementParams.standDistance;
        if (distance <= treeThinker.engagementParams.moveDistance) {
            curPath.movementOptions = {true, false, true};
            curPath.shootOptions = PathShootOptions::Spray;
        }
        else if (distance <= treeThinker.engagementParams.sprayDistance) {
            curPath.movementOptions = {false, false, shouldCrouch};
            curPath.shootOptions = PathShootOptions::Spray;
        }
        else if (distance <= treeThinker.engagementParams.burstDistance) {
            curPath.movementOptions = {false, false, shouldCrouch};
            curPath.shootOptions = PathShootOptions::Burst;
        }
        else {
            curPath.movementOptions = {false, false, shouldCrouch};
            curPath.shootOptions = PathShootOptions::Tap;
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
