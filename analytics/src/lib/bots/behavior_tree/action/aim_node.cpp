//
// Created by durst on 5/8/22.
//

#include "bots/behavior_tree/action_node.h"
#define MAX_LOOK_AT_C4_DISTANCE 300.
#define K_P 0.0025
#define K_I 0.
#define K_D 0.

namespace action {

    float computeAngleVelocityPOnly(double totalDeltaAngle, double lastDeltaAngle) {
        double newDeltaAngle = std::max(-1 * MAX_ONE_DIRECTION_ANGLE_VEL,
                                        std::min(MAX_ONE_DIRECTION_ANGLE_VEL, totalDeltaAngle / 3));
        double newAccelAngle = newDeltaAngle - lastDeltaAngle;
        if (std::abs(newAccelAngle) > MAX_ONE_DIRECTION_ANGLE_ACCEL) {
            newDeltaAngle = lastDeltaAngle +
                            copysign(MAX_ONE_DIRECTION_ANGLE_ACCEL, newAccelAngle);
        }
        return newDeltaAngle / MAX_ONE_DIRECTION_ANGLE_VEL;
    }

    float computeAngleVelocityPID(double deltaAngle, PIDState pidState, double error) {
        // compute P in PID
        double P;
        if (deltaAngle > 3.0) {
            P = deltaAngle * 0.001;
        }
        else if (deltaAngle > 0.5) {
            P = deltaAngle * 0.0055;
        }
        else {
            P = deltaAngle * 0.0015;
        }

        // compute I in PID
        pidState.errorHistory.enqueue(deltaAngle, true);
        double errorSum = 0.;
        for (double error : pidState.errorHistory.getVector()) {
            errorSum += error;
        }
        double I = errorSum / pidState.errorHistory.getCurSize() * K_I;

        // compute D in PID
        double D = (pidState.errorHistory.fromNewest(0) - pidState.errorHistory.fromNewest(1)) * K_D;

        return error*(P + I + D);
    }

    NodeState AimTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        Action & curAction = blackboard.playerToAction[treeThinker.csgoId];
        Action & oldAction = blackboard.lastPlayerToAction[treeThinker.csgoId];
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
        curAction.inputAngleDeltaPctX = computeAngleVelocityPID(deltaAngles.x, blackboard.playerToPIDStateX[treeThinker.csgoId], blackboard.aimDis(blackboard.gen));
        curAction.inputAngleDeltaPctY = computeAngleVelocityPID(deltaAngles.y, blackboard.playerToPIDStateY[treeThinker.csgoId], blackboard.aimDis(blackboard.gen));

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}