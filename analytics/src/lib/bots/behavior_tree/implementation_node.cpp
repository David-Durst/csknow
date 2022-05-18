//
// Created by durst on 5/8/22.
//

#include "bots/behavior_tree/implementation_node.h"
#include "bots/input_bits.h"

namespace implementation {
    NodeState PathingTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        Vec3 curPos = curClient.getFootPosForPlayer();
        const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPos));

        // if have a path and haven't changed nav area id
        // check if priority's nav area is same. If so, do nothing (except increment waypoint if necessary)
        // also check that in nav area you expect to be in from path
        if (blackboard.playerToPath.find(treeThinker.csgoId) != blackboard.playerToPath.end() &&
            blackboard.playerToCurNavAreaId.find(treeThinker.csgoId) != blackboard.playerToCurNavAreaId.end() &&
            curArea.get_id() == blackboard.playerToCurNavAreaId[treeThinker.csgoId]) {
            Path & curPath = blackboard.playerToPath[treeThinker.csgoId];

            if (curPath.waypoints[curPath.curWaypoint].area1 == curArea.get_id() ||
                    curPath.waypoints[curPath.curWaypoint].area2 == curArea.get_id()) {
                // check if player is in a nav mesh
                // fast is point based
                // slow is checking overlaps
                /*
                bool onPath = curPath.areas.find(curArea.get_id()) != curPath.areas.end();
                if (!onPath) {
                    AABB playerAABB = getAABBForPlayer(curPos);
                    for (const auto & pathAreaId : curPath.areas) {
                        const nav_mesh::nav_area & pathArea = blackboard.navFile.get_area_by_id_fast(pathAreaId);
                        AABB navAreaAABB = {vec3tConv(pathArea.get_min_corner()), vec3tConv(pathArea.get_max_corner())};
                        if (aabbOverlap(playerAABB, navAreaAABB)) {
                            onPath = true;
                            break;
                        }
                    }
                }
                 */
                bool onPath = true;
                if (curPath.pathCallSucceeded && onPath) {
                    PathNode curNode = curPath.waypoints[curPath.curWaypoint];
                    Vec3 targetPos = curNode.pos;
                    // ignore z since slope doesn't really matter
                    curPos.z = 0.;
                    targetPos.z = 0.;
                    // ok if in the next area and above it
                    bool areasDisjoint = false;
                    if (curNode.edgeMidpoint) {
                        const nav_mesh::nav_area &priorArea = blackboard.navFile.get_area_by_id_fast(curNode.area1);
                        const nav_mesh::nav_area &nextArea = blackboard.navFile.get_area_by_id_fast(curNode.area2);
                        areasDisjoint = !aabbOverlap(areaToAABB(priorArea), areaToAABB(nextArea));
                        //aboveNextNode = nextArea.is_within(vec3Conv(curPos)) && nextArea.get_max_corner().z < curClient.lastFootPosZ;
                    }
                    // either you are in the navmesh that is the current target, you've entered the target nav mesh of an shared edge
                    // or you are in between two nav meshes that don't share an edge and just need to be close enough
                    // assuming that disjoint areas are mostly free space around them so can't get stuck in x/y coordinates
                    //computeDistance(curPos, targetPos) < MIN_DISTANCE_TO_NAV_POINT &&
                    if ((!curNode.edgeMidpoint && curArea.get_id() == curNode.area1) ||
                        (curNode.edgeMidpoint && curArea.get_id() == curNode.area2) ||
                        (areasDisjoint && computeDistance(curPos, targetPos) < MIN_DISTANCE_TO_NAV_POINT)) {
                        if (curPath.curWaypoint < curPath.waypoints.size() - 1) {
                            curPath.curWaypoint++;
                        }
                    }

                    playerNodeState[treeThinker.csgoId] = NodeState::Running;
                    return playerNodeState[treeThinker.csgoId];
                }
            }
        }

        // otherwise, either no old path or old path is out of date, so update it
        Path newPath;
        auto optionalWaypoints = blackboard.navFile.find_path_detailed(vec3Conv(curClient.getFootPosForPlayer()),
                                                              vec3Conv(curPriority.targetPos));
        if (optionalWaypoints) {
            newPath.pathCallSucceeded = true;
            vector<nav_mesh::PathNode> tmpWaypoints = optionalWaypoints.value();
            for (const auto & tmpWaypoint : tmpWaypoints) {
                newPath.waypoints.push_back(tmpWaypoint);
                newPath.areas.insert(tmpWaypoint.area1);
                if (tmpWaypoint.edgeMidpoint) {
                    newPath.areas.insert(tmpWaypoint.area2);
                }
            }
            newPath.curWaypoint = 0;
            newPath.pathEndAreaId =
                    blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPriority.targetPos)).get_id();

            newPath.movementOptions = {true, false, false};
            newPath.shootOptions = PathShootOptions::DontShoot;
        }
        else {
            // do nothing if the pathing call failed
            newPath.pathCallSucceeded = false;
            blackboard.navFile.find_path_detailed(vec3Conv(curClient.getFootPosForPlayer()),
                                                                           vec3Conv(curPriority.targetPos));
        }
        blackboard.playerToPath[treeThinker.csgoId] = newPath;

        blackboard.playerToCurNavAreaId[treeThinker.csgoId] = curArea.get_id();
        playerNodeState[treeThinker.csgoId] = newPath.pathCallSucceeded ? NodeState::Running : NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
    }

    NodeState FireSelectionTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
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

