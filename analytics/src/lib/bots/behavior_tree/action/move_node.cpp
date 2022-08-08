//
// Created by durst on 5/8/22.
//

#include "bots/input_bits.h"
#include "bots/behavior_tree/action/action_node.h"
#include "bots/behavior_tree/pathing_node.h"

namespace action {
    void stop(Action & curAction) {
        // don't need to worry about targetAngles y since can't move up and down
        curAction.setButton(IN_FORWARD, false);
        curAction.setButton(IN_MOVELEFT, false);
        curAction.setButton(IN_BACK, false);
        curAction.setButton(IN_MOVERIGHT, false);
    }

    void moveInDir(Action & curAction, Vec2 dir) {
        // don't need to worry about targetAngles y since can't move up and down
        curAction.setButton(IN_FORWARD, dir.x >= -80. && dir.x <= 80.);
        curAction.setButton(IN_MOVELEFT, dir.x >= 10. && dir.x <= 170.);
        curAction.setButton(IN_BACK, dir.x >= 100. || dir.x <= -100.);
        curAction.setButton(IN_MOVERIGHT, dir.x >= -170. && dir.x <= -10.);
    }

    NodeState MovementTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        Path & curPath = blackboard.playerToPath[treeThinker.csgoId];
        Action & curAction = blackboard.playerToAction[treeThinker.csgoId];
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        Vec3 curPos = curClient.getFootPosForPlayer();
        const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPos));

        if (curPath.pathCallSucceeded) {

            // don't move if move is set to false on priority
            if (!curPriority.moveOptions.move) {
                stop(curAction);
                // add counter strafing later
            }
            else {
                Vec2 curViewAngle = curClient.getCurrentViewAnglesWithAimpunch();
                Vec3 targetVector = curPath.waypoints[curPath.curWaypoint].pos - curClient.getEyePosForPlayer();
                // add eye height since waypoints are on floor and aim is from eye
                targetVector.z += EYE_HEIGHT;
                Vec2 targetViewAngle = vectorAngles(targetVector);
                targetViewAngle.makePitchNeg90To90();

                Vec2 deltaViewAngle = targetViewAngle - curViewAngle;
                deltaViewAngle.makeYawNeg180To180();
                moveInDir(curAction, deltaViewAngle);

                const PathNode & curNode = curPath.waypoints[curPath.curWaypoint];
                // jump if moving to higher navmesh area, cur area, not marked no jump, and near target navmesh
                const nav_mesh::nav_area & dstArea = curNode.edgeMidpoint ? blackboard.navFile.get_area_by_id_fast(curNode.area2) :
                                                     blackboard.navFile.get_area_by_id_fast(curNode.area1);
                if (curArea.get_id() == 8654 || curArea.get_id() == 6953 || curArea.get_id() == 6802) {
                    int x = 1;
                }
                if (curArea.get_id() == 9107 || curArea.get_id() == 9108) {
                    int x = 1;
                }

                // can't compare current nav area to target nav area as current nav area max z different from current pos z
                // (see d2 slope to A site)
                if (blackboard.navFile.get_point_to_area_distance_2d(vec3Conv(curPos), dstArea) < 40. &&
                    dstArea.get_min_corner().z > curPos.z + MAX_OBSTACLE_SIZE) {
                    // make sure moving into target in 2d
                    // check if aiming at enemy anywhere
                    /*
                     * // for now, assume moving in correct direction, but in future may want a velocity check
                    Vec3 footPos = curClient.getFootPosForPlayer(), dir = targetVector;
                    footPos.z = 0;
                    dir.z = 0;
                    Ray source{footPos, dir};

                    AABB targetAABB{vec3tConv(dstArea.m_nw_corner), vec3tConv(dstArea.m_se_corner)};
                    targetAABB.min.z = -10.;
                    targetAABB.max.z = 10.;
                    double hitt0, hitt1;
                    bool movingToDst = intersectP(targetAABB, source, hitt0, hitt1);
                    */

                    // make sure near target navmesh
                    bool closeToDst = blackboard.navFile.get_point_to_area_distance(vec3Conv(curClient.getFootPosForPlayer()), dstArea) < 100.;
                    bool jumpResetTimePassed = state.getSecondsBetweenTimes(curAction.lastJumpTime, state.loadTime) > MIN_JUMP_RESET_SECONDS;
                    bool shouldJump = closeToDst && jumpResetTimePassed && curClient.lastVelX < 2. && curClient.lastVelY < 2.;

                    if (shouldJump) {
                        curAction.lastJumpTime = state.loadTime;
                    }
                    curAction.setButton(IN_JUMP, shouldJump);
                }
            }

            // regardless if moving, check for crouching
            curAction.setButton(IN_WALK, curPriority.moveOptions.walk);
            // always crouch when airborne to get to max area
            curAction.setButton(IN_DUCK, curPriority.moveOptions.crouch || curClient.isAirborne);

        }
        // do nothing if there was an error
        else {
            curAction.setButton(IN_FORWARD, false);
            curAction.setButton(IN_MOVELEFT, false);
            curAction.setButton(IN_BACK, false);
            curAction.setButton(IN_MOVERIGHT, false);
            curAction.inputAngleDeltaPctX = 0;
            curAction.inputAngleDeltaPctY = 0;
        }

        playerNodeState[treeThinker.csgoId] = curPath.pathCallSucceeded ? NodeState::Success : NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
    }
}
