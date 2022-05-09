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
        const Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        Vec3 aimTarget;

        // if no visible enemy to target, look at c4 (if planted and close) or next location
        if (curPriority.priorityType != PriorityType::Player) {
            if (curPriority.priorityType == PriorityType::C4 && state.c4IsPlanted &&
                computeDistance(curClient.getFootPosForPlayer(), state.getC4Pos()) <= MAX_LOOK_AT_C4_DISTANCE) {
                aimTarget = state.getC4Pos();
            }
            else {

            }
        }

    }

}



void Thinker::aimAt(ServerState::Client & curClient, const ServerState::Client & targetClient,
                    const Vec2 & aimOffset, const ServerState::Client & priorClient) {
    // if no plant and no enemies, then nothing to look at
    if (targetClient.csgoId == INVALID_ID && !liveState.c4IsPlanted) {
        curClient.inputAngleDeltaPctX = 0.;
        curClient.inputAngleDeltaPctY = 0.;
        return;
    }

    // look at an enemy if it exists, otherwise look at bomb
    Vec3 targetVector;
    if (targetClient.csgoId != INVALID_ID) {
        targetVector = {
                targetClient.lastEyePosX - curClient.lastEyePosX,
                targetClient.lastEyePosY - curClient.lastEyePosY,
                targetClient.lastEyePosZ - curClient.lastEyePosZ
        };
        Vec3 targetView = angleVectors({curClient.lastEyeWithRecoilAngleX,
                                        curClient.lastEyeWithRecoilAngleY});
        targetVector = targetVector + targetView * HEAD_ADJUSTMENT;
    }
    else {
        targetVector = {
                liveState.c4X - curClient.lastEyePosX,
                liveState.c4Y - curClient.lastEyePosY,
                liveState.c4Z - curClient.lastEyePosZ
        };
    }

    Vec2 currentAngles = {
            curClient.lastEyeAngleX + curClient.lastAimpunchAngleX,
            curClient.lastEyeAngleY + curClient.lastAimpunchAngleY};
    Vec2 targetAngles = vectorAngles(targetVector) + aimOffset;
    targetAngles.makePitchNeg90To90();
    targetAngles.y = std::max(-1 * MAX_PITCH_MAGNITUDE,
                              std::min(MAX_PITCH_MAGNITUDE, targetAngles.y));

    // https://stackoverflow.com/a/7428771
    Vec2 deltaAngles = targetAngles - currentAngles;
    deltaAngles.makeYawNeg180To180();

    curClient.inputAngleDeltaPctX = computeAngleVelocity(deltaAngles.x, priorClient.inputAngleDeltaPctX);
    curClient.inputAngleDeltaPctY = computeAngleVelocity(deltaAngles.y, priorClient.inputAngleDeltaPctY);
}
