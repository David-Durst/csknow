//
// Created by durst on 5/8/22.
//

#include "bots/behavior_tree/action/action_node.h"
#define MAX_LOOK_AT_C4_DISTANCE 300.
#define SECOND_ORDER false
#define K_P 0.0025
#define K_I 0.
#define K_D 0.

namespace action {
    float computeAngleVelocityPID(double deltaAngle, PIDState pidState, double noise) {
        // compute P in PID
        double P;
        //P = deltaAngle * 0.001;
        if (deltaAngle > 30.0) {
            P = deltaAngle * 0.003;
        }
        else if (deltaAngle > 15.0) {
            P = deltaAngle * 0.002;
        }
        else if (deltaAngle > 5.0) {
            P = deltaAngle * 0.002;
        }
        else if (deltaAngle > 5.0) {
            P = deltaAngle * 0.002;
        }
        else if (deltaAngle > 0.5) {
            P = deltaAngle * 0.003;
        }
        else {
            P = deltaAngle * 0.005;
        }

        // compute I in PID
        pidState.errorHistory.enqueue(deltaAngle, true);
        double errorSum = 0.;
        for (double error : pidState.errorHistory.getVector()) {
            errorSum += error;
        }
        double I = errorSum / pidState.errorHistory.getCurSize() * K_I;

        // compute D in PID
        /*
        double K_DT;
        //P = deltaAngle * 0.001;
        if (deltaAngle > 0.1) {
            K_DT = 0;
        }
        else {
            K_DT = 0.005;
        }
         */
        double D = (pidState.errorHistory.fromNewest(0) - pidState.errorHistory.fromOldest(0)) * K_D;

        if (deltaAngle > 0.01) {
            return noise*(P + I + D);
        }
        else {
            return 5 * noise * (P + I + D);
        }
    }

    Vec2 makeAngleToPct(Vec2 deltaAngle) {
        return deltaAngle / MAX_ONE_DIRECTION_ANGLE_VEL;
    }

    NodeState AimTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        SecondOrderController & mouseController = blackboard.playerToMouseController.find(treeThinker.csgoId)->second;
        const Order & curOrder = blackboard.strategy.getOrderForPlayer(treeThinker.csgoId);
        Action & curAction = blackboard.playerToAction[treeThinker.csgoId];
        //Action & oldAction = blackboard.lastPlayerToAction[treeThinker.csgoId];
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        Path & curPath = blackboard.playerToPath[treeThinker.csgoId];
        Vec3 aimTarget;

        if (!curPath.pathCallSucceeded) {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }

        // aim at target player if one exists, or aim at danger location, otherwise the next way point
        // this should handle c4 as end of path will be c4
        if (curPriority.targetPlayer.playerId != INVALID_ID) {
            aimTarget = curPriority.targetPlayer.eyePos;
            curAction.aimTargetType = AimTargetType::Player;
        }
        else if (blackboard.isPlayerDefuser(treeThinker.csgoId) &&
            curOrder.waypoints[blackboard.strategy.playerToWaypointIndex[treeThinker.csgoId]].type == WaypointType::C4) {
            aimTarget = state.getC4Pos();
            curAction.aimTargetType = AimTargetType::C4;
        }
        else if (curPriority.nonDangerAimArea && curPriority.nonDangerAimAreaType == NonDangerAimAreaType::Hold) {
            aimTarget = vec3tConv(blackboard.navFile.get_area_by_id_fast(curPriority.nonDangerAimArea.value()).get_center());
            aimTarget.z += EYE_HEIGHT;
            curAction.aimTargetType = AimTargetType::HoldNonDangerArea;
        }
        else if (blackboard.playerToDangerAreaId.find(treeThinker.csgoId) != blackboard.playerToDangerAreaId.end()) {
            aimTarget = vec3tConv(blackboard.navFile.get_area_by_id_fast(blackboard.playerToDangerAreaId[treeThinker.csgoId]).get_center());
            aimTarget.z += EYE_HEIGHT;
            curAction.aimTargetType = AimTargetType::DangerArea;
        }
        else if (curPriority.nonDangerAimArea && curPriority.nonDangerAimAreaType == NonDangerAimAreaType::Path) {
            aimTarget = vec3tConv(blackboard.navFile.get_area_by_id_fast(curPriority.nonDangerAimArea.value()).get_center());
            aimTarget.z += EYE_HEIGHT;
            curAction.aimTargetType = AimTargetType::PathNonDangerArea;
        }
        else {
            aimTarget = curPath.waypoints[curPath.curWaypoint].pos;
            aimTarget.z += EYE_HEIGHT;
            curAction.aimTargetType = AimTargetType::Waypoint;
        }

        // save target for learned model
        csknow::engagement_aim::ClientTargetMap & clientTargetMap = blackboard.streamingManager.streamingEngagementAim
            .currentClientTargetMap;
        if (curAction.aimTargetType == AimTargetType::Player) {
            clientTargetMap[curClient.csgoId] = {curPriority.targetPlayer.playerId, {0., 0., 0.}};
        }
        else {
            clientTargetMap[curClient.csgoId] = {INVALID_ID, aimTarget};
        }


        // don't need to change pitch here because engine stores pitch in -90 to 90 (I think)
        // while conversion function below uses 360-270 for -90-0
        Vec2 curViewAngle = curClient.getCurrentViewAnglesWithAimpunch();
        Vec3 targetVector = aimTarget - curClient.getEyePosForPlayer();
        Vec2 targetViewAngle = vectorAngles(targetVector);

        targetViewAngle.makePitchNeg90To90();

        // https://stackoverflow.com/a/7428771
        Vec2 deltaAngle = targetViewAngle - curViewAngle;
        deltaAngle.makeYawNeg180To180();

        if (SECOND_ORDER) {
            Vec2 newDeltaAngle = mouseController.update(state.getSecondsBetweenTimes(curAction.lastActionTime, state.loadTime),
                                                        deltaAngle, {0., 0.});
            Vec2 newDeltaAnglePct = makeAngleToPct(newDeltaAngle);
            curAction.inputAngleX = newDeltaAnglePct.x;
            curAction.inputAngleY = newDeltaAnglePct.y;
            /*
            double velocity = std::abs(computeMagnitude(newDeltaAnglePct));
            curAction.rollingAvgMouseVelocity = curAction.rollingAvgMouseVelocity * 0.5 + velocity * 0.5;
            double absAccel = std::abs(velocity  - curAction.rollingAvgMouseVelocity);
            if (absAccel < 0.01 && velocity < 0.01 && mouseController.isYDReset()) {
                mouseController.reset();
                // do computation a second time with reset values
                Vec2 newDeltaAnglePct = makeAngleToPct(newDeltaAngle);
                curAction.inputAngleX = newDeltaAnglePct.x;
                curAction.inputAngleY = newDeltaAnglePct.y;
                curAction.enableSecondOrder = true;
            }
            else if (absAccel > 0.4 && velocity > 0.9) {
                //curAction.enableSecondOrder = false;
            }
            curAction.enableSecondOrder = true;
             */
            curAction.lastActionTime = state.loadTime;
        }
        else {
            const unordered_map<CSGOId, Vec2> & playerToNewAngle =
                blackboard.streamingManager.streamingEngagementAim.playerToNewAngle;
            if (playerToNewAngle.find(curClient.csgoId) == playerToNewAngle.end()) {
                curAction.inputAngleX = 0.;
                curAction.inputAngleY = 0.;
            }
            else {
                Vec2 newAngle = playerToNewAngle.at(curClient.csgoId);
                curAction.inputAngleX = newAngle.x;
                curAction.inputAngleY = newAngle.y;
                if (curClient.csgoId == 3 && false) {
                    std::cout << curClient.name << " (" << curClient.csgoId << ") "
                        << "frame: " << curClient.lastFrame
                        << "cur view angle: " << curClient.getCurrentViewAngles().toString()
                        << "new view angle: " << newAngle.toString()
                        << std::endl;
                }
            }
        }

        /*
        if (!SECOND_ORDER || !curAction.enableSecondOrder) { //|| computeMagnitude(deltaAngle) > 40) {
            // TODO: use better angle velocity control
            curAction.inputAngleX = computeAngleVelocityPID(deltaAngle.x, blackboard.playerToPIDStateX[treeThinker.csgoId], blackboard.aimDis(blackboard.gen));
            curAction.inputAngleY = computeAngleVelocityPID(deltaAngle.y, blackboard.playerToPIDStateY[treeThinker.csgoId], blackboard.aimDis(blackboard.gen));

            curAction.inputAngleX = curAction.inputAngleX * 0.45 + oldAction.inputAngleX * 0.55;
            curAction.inputAngleY = curAction.inputAngleY * 0.45 + oldAction.inputAngleY * 0.55;

        }
         */

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}