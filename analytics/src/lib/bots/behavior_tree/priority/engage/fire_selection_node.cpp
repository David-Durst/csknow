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
            curPriority.moveOptions = {true, false, false};
            curPriority.shootOptions = ShootOptions::DontShoot;

            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }

        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        const ServerState::Client & targetClient = state.getClient(curPriority.targetPlayer.playerId);
        double distance = computeDistance(curClient.getFootPosForPlayer(), targetClient.getFootPosForPlayer());

        // if close enough to move and shoot, crouch
        bool shouldCrouch = distance <= treeThinker.engagementParams.standDistance;
        if (!curPriority.targetPlayer.visible) {
            curPriority.moveOptions = {true, false, true};
            curPriority.shootOptions = ShootOptions::DontShoot;
        }
        else if (distance <= treeThinker.engagementParams.moveDistance) {
            curPriority.moveOptions = {true, false, true};
            curPriority.shootOptions = ShootOptions::Spray;
        }
        else if (distance <= treeThinker.engagementParams.sprayDistance) {
            curPriority.moveOptions = {false, false, shouldCrouch};
            curPriority.shootOptions = ShootOptions::Spray;
        }
        else if (distance <= treeThinker.engagementParams.burstDistance) {
            curPriority.moveOptions = {false, false, shouldCrouch};
            curPriority.shootOptions = ShootOptions::Burst;
        }
        else {
            curPriority.moveOptions = {false, false, shouldCrouch};
            curPriority.shootOptions = ShootOptions::Tap;
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
