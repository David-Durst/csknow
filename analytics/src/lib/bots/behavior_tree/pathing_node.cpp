//
// Created by durst on 5/8/22.
//

#include "bots/input_bits.h"
#include "bots/behavior_tree/pathing_node.h"

namespace movement {
    /*
    void updateAreasToIncreaseCost(const ServerState &state, Blackboard & blackboard, const ServerState::Client & curClient) {
        set<uint32_t> areasToIncreaseCost;
        uint32_t curArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curClient.getFootPosForPlayer())).get_id();
        for (const auto & client : state.clients) {
            if (client.isAlive && client.csgoId != curClient.csgoId && client.team == curClient.team) {
                uint32_t otherArea = blackboard.navFile
                        .get_nearest_area_by_position(vec3Conv(client.getFootPosForPlayer())).get_id();
                if (curArea != otherArea) {
                    areasToIncreaseCost.insert(otherArea);
                }
            }
        }
        blackboard.navFile.set_areas_to_increase_cost(areasToIncreaseCost);
    }
     */

    Path computePath(Blackboard & blackboard, nav_mesh::vec3_t preCheckTargetPos,
                     const ServerState::Client & curClient) {
        Path newPath;
        //updateAreasToIncreaseCost(state, blackboard, curClient);
        AreaId targetAreaId = blackboard.navFile.get_nearest_area_by_position(preCheckTargetPos).get_id();
        nav_mesh::vec3_t targetPos;
        if (blackboard.removedAreas.find(targetAreaId) != blackboard.removedAreas.end()) {
            targetPos = blackboard.navFile.get_area_by_id_fast(blackboard.removedAreaAlternatives[targetAreaId]).get_center();
        }
        else {
            targetPos = preCheckTargetPos;
        }

        auto optionalWaypoints =
                blackboard.navFile.find_path_detailed(vec3Conv(curClient.getFootPosForPlayer()), targetPos);
        if (optionalWaypoints) {
            newPath.pathCallSucceeded = true;
            vector<nav_mesh::PathNode> tmpWaypoints = optionalWaypoints.value();
            for (const auto & tmpWaypoint : tmpWaypoints) {
                newPath.waypoints.emplace_back(tmpWaypoint);
                newPath.areas.insert(tmpWaypoint.area1);
                if (tmpWaypoint.edgeMidpoint) {
                    newPath.areas.insert(tmpWaypoint.area2);
                }
            }
            newPath.curWaypoint = 0;
            newPath.pathEndAreaId =
                    blackboard.navFile.get_nearest_area_by_position(targetPos).get_id();
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
        uint32_t targetAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPriority.targetPos)).get_id();

        // if have a path and haven't changed source or target nav area id and not stuck.
        // If so, do nothing (except increment waypoint if necessary)
        // also check that not in an already visited area because missed a jump
        if (blackboard.playerToPath.find(treeThinker.csgoId) != blackboard.playerToPath.end() &&
            blackboard.playerToLastPathingSourceNavAreaId.find(treeThinker.csgoId) != blackboard.playerToLastPathingSourceNavAreaId.end() &&
            curArea.get_id() == blackboard.playerToLastPathingSourceNavAreaId[treeThinker.csgoId] &&
            blackboard.playerToLastPathingTargetNavAreaId.find(treeThinker.csgoId) != blackboard.playerToLastPathingTargetNavAreaId.end() &&
            targetAreaId == blackboard.playerToLastPathingTargetNavAreaId[treeThinker.csgoId]) {
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
                    AABB curAABB = areaToAABB(priorArea), nextAABB = areaToAABB(nextArea);
                    // overlap big enough that small AABB on B hole connect, but small enough that ledge on A to A ramp
                    // doesn't
                    curAABB.expand(OVERLAP_SIZE);
                    areasDisjoint = !aabbOverlap(curAABB, nextAABB);
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
        Path newPath = computePath(blackboard, vec3Conv(curPriority.targetPos), curClient);
        /*
        // IF LOOPING, LOOK HERE FOR CYCLES
        if (blackboard.playerToPath.find(treeThinker.csgoId) != blackboard.playerToPath.end()) {
            Path oldPath = blackboard.playerToPath[treeThinker.csgoId];
            if (oldPath.waypoints[0].area1 == newPath.waypoints[2].area1) {
                double z = blackboard.navFile.get_point_to_area_distance(vec3Conv(curClient.getFootPosForPlayer()), blackboard.navFile.get_area_by_id_fast(oldPath.waypoints[0].area1));
                double y = blackboard.navFile.get_point_to_area_distance(vec3Conv(curClient.getFootPosForPlayer()), blackboard.navFile.get_area_by_id_fast(newPath.waypoints[0].area1));
                const nav_mesh::nav_area & curArea = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPos));
            }
        }
         */
        blackboard.playerToPath[treeThinker.csgoId] = newPath;

        blackboard.playerToLastPathingSourceNavAreaId[treeThinker.csgoId] = curArea.get_id();
        blackboard.playerToLastPathingTargetNavAreaId[treeThinker.csgoId] = targetAreaId;
        playerNodeState[treeThinker.csgoId] = newPath.pathCallSucceeded ? NodeState::Success : NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
    }

    NodeState WaitNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        if (playerNodeState[treeThinker.csgoId] == NodeState::Uninitialized) {
            startTime[treeThinker.csgoId] = state.loadTime;
        }

        double timeSinceStart = ServerState::getSecondsBetweenTimes(startTime[treeThinker.csgoId], state.loadTime);
        if (timeSinceStart >= waitSeconds) {
            playerNodeState[treeThinker.csgoId] = succeedOnEnd ? NodeState::Success : NodeState::Failure;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
        }
        return playerNodeState[treeThinker.csgoId];
    }

    NodeState WaitTicksNode::exec(const ServerState &, TreeThinker &treeThinker) {
        if (playerNodeState[treeThinker.csgoId] == NodeState::Uninitialized) {
            numTicksWaited[treeThinker.csgoId] = 0;
        }
        else {
            numTicksWaited[treeThinker.csgoId]++;
        }

        if (numTicksWaited[treeThinker.csgoId] >= waitTicks) {
            playerNodeState[treeThinker.csgoId] = succeedOnEnd ? NodeState::Success : NodeState::Failure;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Running;
        }
        return playerNodeState[treeThinker.csgoId];
    }

}

