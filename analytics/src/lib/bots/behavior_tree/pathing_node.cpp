//
// Created by durst on 5/8/22.
//

#include "bots/input_bits.h"
#include "bots/behavior_tree/pathing_node.h"

namespace movement {
    Path computePath(const ServerState &state, Blackboard & blackboard, nav_mesh::vec3_t targetPos, const ServerState::Client & curClient) {
        Path newPath;
        auto optionalWaypoints = blackboard.navFile.find_path_detailed(vec3Conv(curClient.getFootPosForPlayer()),
                                                                       targetPos);
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
                    blackboard.navFile.get_nearest_area_by_position(targetPos).get_id();

            newPath.movementOptions = {true, false, false};
            newPath.shootOptions = PathShootOptions::DontShoot;
        }
        else {
            // do nothing if the pathing call failed
            newPath.pathCallSucceeded = false;
            blackboard.navFile.find_path_detailed(vec3Conv(curClient.getFootPosForPlayer()), targetPos);
        }
        return newPath;
    }

    NodeState PathingNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        Vec3 curPos = curClient.getFootPosForPlayer();
        const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPos));

        // if have a path and haven't changed nav area id
        // check if players's nav area is same and not stuck. If so, do nothing (except increment waypoint if necessary)
        // also check that not in an already visited area because missed a jump
        if (blackboard.playerToPath.find(treeThinker.csgoId) != blackboard.playerToPath.end() &&
            blackboard.playerToLastPathingNavAreaId.find(treeThinker.csgoId) != blackboard.playerToLastPathingNavAreaId.end() &&
            curArea.get_id() == blackboard.playerToLastPathingNavAreaId[treeThinker.csgoId] &&
            !curPriority.stuck) {
            Path & curPath = blackboard.playerToPath[treeThinker.csgoId];

            if (curPath.pathCallSucceeded) {
                PathNode targetNode = curPath.waypoints[curPath.curWaypoint];
                Vec3 targetPos = targetNode.pos;
                // ignore z since slope doesn't really matter
                curPos.z = 0.;
                targetPos.z = 0.;
                // ok if in the next area and above it
                bool areasDisjoint = false;
                if (targetNode.edgeMidpoint) {
                    const nav_mesh::nav_area &priorArea = blackboard.navFile.get_area_by_id_fast(targetNode.area1);
                    const nav_mesh::nav_area &nextArea = blackboard.navFile.get_area_by_id_fast(targetNode.area2);
                    areasDisjoint = !aabbOverlap(areaToAABB(priorArea), areaToAABB(nextArea));
                    //aboveNextNode = nextArea.is_within(vec3Conv(curPos)) && nextArea.get_max_corner().z < curClient.lastFootPosZ;
                }
                // either you are in the navmesh that is the current target, you've entered the target nav mesh of an shared edge
                // or you are in between two nav meshes that don't share an edge and just need to be close enough
                // assuming that disjoint areas are mostly free space around them so can't get stuck in x/y coordinates
                //computeDistance(curPos, targetPos) < MIN_DISTANCE_TO_NAV_POINT &&
                if ((!targetNode.edgeMidpoint && curArea.get_id() == targetNode.area1) ||
                    (targetNode.edgeMidpoint && curArea.get_id() == targetNode.area2) ||
                    (areasDisjoint && computeDistance(curPos, targetPos) < MIN_DISTANCE_TO_NAV_POINT)) {
                    if (curPath.curWaypoint < curPath.waypoints.size() - 1) {
                        curPath.curWaypoint++;
                    }
                }

                playerNodeState[treeThinker.csgoId] = NodeState::Success;
                return playerNodeState[treeThinker.csgoId];
            }
        }

        // otherwise, either no old path or old path is out of date, so update it
        Path newPath = computePath(state, blackboard, vec3Conv(curPriority.targetPos), curClient);
        blackboard.playerToPath[treeThinker.csgoId] = newPath;
        if (newPath.areas.find(9026) != newPath.areas.end() && curPriority.stuck) {
            int x = 1;
        }

        blackboard.playerToLastPathingNavAreaId[treeThinker.csgoId] = curArea.get_id();
        playerNodeState[treeThinker.csgoId] = newPath.pathCallSucceeded ? NodeState::Success : NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
    }

    NodeState WaitNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);

        if (playerNodeState[treeThinker.csgoId] == NodeState::Uninitialized) {
            startFrame[treeThinker.csgoId] = curClient.lastFrame;
        }

        double timeSinceStart = (curClient.lastFrame - startFrame[treeThinker.csgoId]) * state.tickInterval;
        if (timeSinceStart >= waitSeconds) {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
        }
        return playerNodeState[treeThinker.csgoId];
    }

}

