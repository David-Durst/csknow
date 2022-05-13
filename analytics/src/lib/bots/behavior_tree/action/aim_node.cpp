//
// Created by durst on 5/8/22.
//

#include "bots/behavior_tree/action_node.h"
#define MAX_LOOK_AT_C4_DISTANCE 300.

namespace action {

    float computeAngleVelocity(double totalDeltaAngle, double lastDeltaAngle) {
        double newDeltaAngle = std::max(-1 * MAX_ONE_DIRECTION_ANGLE_VEL,
                                        std::min(MAX_ONE_DIRECTION_ANGLE_VEL, totalDeltaAngle / 3));
        double newAccelAngle = newDeltaAngle - lastDeltaAngle;
        if (std::abs(newAccelAngle) > MAX_ONE_DIRECTION_ANGLE_ACCEL) {
            newDeltaAngle = lastDeltaAngle +
                            copysign(MAX_ONE_DIRECTION_ANGLE_ACCEL, newAccelAngle);
        }
        return newDeltaAngle / MAX_ONE_DIRECTION_ANGLE_VEL;
    }

    NodeState AimTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        Action & curAction = blackboard.playerToAction[treeThinker.csgoId];
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        Path & curPath = blackboard.playerToPath[treeThinker.csgoId];
        Vec3 aimTarget;

        if (!curPath.pathCallSucceeded) {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }

        // aim at target player if one exists, otherwise the next way point
        // this should handle c4 as end of path will be c4
        if (curPriority.targetPlayer.playerId == INVALID_ID) {
            aimTarget = curPath.waypoints[curPath.curWaypoint].pos;
            aimTarget.z += EYE_HEIGHT;
        }
        else {
            aimTarget = state.getClient(curPriority.targetPlayer.playerId).getEyePosForPlayer();
        }

        Vec3 targetVector = aimTarget - curClient.getEyePosForPlayer();
        Vec2 targetViewAngle = vectorAngles(targetVector);
        targetViewAngle.makePitchNeg90To90();
        // clamp within max of -89 to 89
        targetViewAngle.y = std::max(-1 * MAX_PITCH_MAGNITUDE,
                                  std::min(MAX_PITCH_MAGNITUDE, targetViewAngle.y));

        // https://stackoverflow.com/a/7428771
        Vec2 deltaAngles = targetViewAngle - curClient.getCurrentViewAnglesWithAimpunch();
        deltaAngles.makeYawNeg180To180();

        // TODO: use better angle velocity control
        curAction.inputAngleDeltaPctX = computeAngleVelocity(deltaAngles.x, curAction.inputAngleDeltaPctX);
        curAction.inputAngleDeltaPctY = computeAngleVelocity(deltaAngles.y, curAction.inputAngleDeltaPctY);

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}