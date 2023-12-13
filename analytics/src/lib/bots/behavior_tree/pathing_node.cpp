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
                     const ServerState::Client & curClient, std::optional<Vec3> c4Pos) {
        Path newPath;
        //updateAreasToIncreaseCost(state, blackboard, curClient);
        AreaId targetAreaId = blackboard.navFile.get_nearest_area_by_position(preCheckTargetPos).get_id();
        nav_mesh::vec3_t targetPos;
        bool validArea = false;
        if (blackboard.removedAreas.find(targetAreaId) != blackboard.removedAreas.end()) {
            targetPos = blackboard.navFile.get_area_by_id_fast(blackboard.removedAreaAlternatives[targetAreaId]).get_center();
        }
        else {
            validArea = true;
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
            double pathDistance = computeDistance(curClient.getFootPosForPlayer(), vec3tConv(targetPos));
            if (validArea && (!c4Pos || vec3Conv(c4Pos.value()) != targetPos) && pathDistance > 4*WIDTH) {
                size_t targetAreaIndex = blackboard.navFile.m_area_ids_to_indices.at(targetAreaId);
                const AABB & targetAreaAABB = blackboard.reachability.coordinate[targetAreaIndex];
                // create an extra waypoint that is known to be good before getting to end
                Vec3 & endPos = newPath.waypoints.back().pos;
                //newPath.waypoints.push_back(newPath.waypoints.back());
                // force player to center if not wide enough
                if (targetAreaAABB.max.x - targetAreaAABB.min.x < WIDTH*4) {
                    endPos.x = getCenter(targetAreaAABB).x;
                }
                else {
                    endPos.x = std::min(targetAreaAABB.max.x - WIDTH * 2, endPos.x);
                    endPos.x = std::max(targetAreaAABB.min.x + WIDTH * 2, endPos.x);
                }
                if (targetAreaAABB.max.y - targetAreaAABB.min.y < WIDTH*4) {
                    endPos.y = getCenter(targetAreaAABB).y;
                }
                else {
                    endPos.y = std::min(targetAreaAABB.max.y - WIDTH * 2, endPos.y);
                    endPos.y = std::max(targetAreaAABB.min.y + WIDTH * 2, endPos.y);
                }
            }
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
        if (blackboard.forceTestPosPerPlayer.find(curClient.csgoId) != blackboard.forceTestPosPerPlayer.end()) {
            targetAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(
                    blackboard.forceTestPosPerPlayer.at(curClient.csgoId))).get_id();
        }

        // filter out accidental cur area changes that are recoverable
        bool changeCurArea = curArea.get_id() != blackboard.playerToLastPathingSourceNavAreaId[treeThinker.csgoId];
        if (changeCurArea && blackboard.playerToPath.find(treeThinker.csgoId) != blackboard.playerToPath.end() &&
            blackboard.playerToLastPathingSourceNavAreaId.find(treeThinker.csgoId) != blackboard.playerToLastPathingSourceNavAreaId.end()) {
            Path & curPath = blackboard.playerToPath[treeThinker.csgoId];
            if (curPath.areas.find(curArea.get_id()) == curPath.areas.end()) {
                for (const auto & conId : curArea.get_connections()) {
                    if (blackboard.playerToLastPathingSourceNavAreaId[treeThinker.csgoId] == conId.id) {
                        changeCurArea = false;
                        break;
                    }
                }
            }
        }

        bool newDeltaPosDestination = blackboard.inferenceManager.ranDeltaPosInferenceThisTick;

        // if have a path and haven't changed source or target nav area id and not stuck.
        // If so, do nothing (except increment waypoint if necessary)
        // also check that not in an already visited area because missed a jump
        if (blackboard.playerToPath.find(treeThinker.csgoId) != blackboard.playerToPath.end() &&
            blackboard.playerToLastPathingSourceNavAreaId.find(treeThinker.csgoId) != blackboard.playerToLastPathingSourceNavAreaId.end() &&
            !changeCurArea && !newDeltaPosDestination &&
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
        Vec3 targetPos = curPriority.targetPos;
        if (blackboard.forceTestPosPerPlayer.find(curClient.csgoId) != blackboard.forceTestPosPerPlayer.end()) {
            targetPos = blackboard.forceTestPosPerPlayer.at(curClient.csgoId);
        }
        Path newPath = computePath(blackboard, vec3Conv(targetPos), curClient, state.getC4Pos());
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

