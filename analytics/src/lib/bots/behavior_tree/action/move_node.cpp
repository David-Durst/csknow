//
// Created by durst on 5/8/22.
//

#include "bots/input_bits.h"
#include "bots/behavior_tree/action_node.h"

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

        if (curPath.pathCallSucceeded) {

            // don't move if move is set to false on priority
            if (!curPath.movementOptions.move) {
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
                if (curNode.edgeMidpoint) {
                    const nav_mesh::nav_area & srcArea = blackboard.navFile.get_area_by_id_fast(curNode.area1);
                    const nav_mesh::nav_area & dstArea = blackboard.navFile.get_area_by_id_fast(curNode.area2);
                    if (!srcArea.is_flag_set(NavAttributeType::NAV_MESH_NO_JUMP) && dstArea.m_nw_corner.z > srcArea.m_se_corner.z) {
                        // make sure moving into target in 2d
                        // check if aiming at enemy anywhere
                        Vec3 footPos = curClient.getFootPosForPlayer(), vel = curClient.getVelocity();
                        footPos.z = 0;
                        vel.z = 0;
                        Ray source{footPos, vel};

                        AABB targetAABB{vec3tConv(dstArea.m_nw_corner), vec3tConv(dstArea.m_se_corner)};
                        targetAABB.min.z = -10.;
                        targetAABB.max.z = 10.;
                        double hitt0, hitt1;
                        bool movingToDst = intersectP(targetAABB, source, hitt0, hitt1);

                        // make sure near target navmesh
                        bool closeToDst = blackboard.navFile.get_point_to_area_distance(vec3Conv(footPos), dstArea) < 20.;

                        curAction.setButton(IN_JUMP, movingToDst && closeToDst);
                    }
                }
            }

            // regardless if moving, check for crouching
            curAction.setButton(IN_WALK, curPath.movementOptions.walk);
            curAction.setButton(IN_DUCK, curPath.movementOptions.crouch);

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

        bool moving = curAction.getButton(IN_FORWARD) ||
                curAction.getButton(IN_MOVELEFT) ||
                curAction.getButton(IN_BACK) ||
                curAction.getButton(IN_MOVERIGHT);

        playerNodeState[treeThinker.csgoId] = moving ? NodeState::Success : NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
    }
}
