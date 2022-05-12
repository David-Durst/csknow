//
// Created by durst on 5/8/22.
//

#include "bots/behavior_tree/implementation_node.h"
#include "bots/input_bits.h"

namespace implementation {
    NodeState PathingTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];

        // check if priority's nav area is same. If so, do nothing (except increment waypoint if necessary)
        if (blackboard.playerToPath.find(treeThinker.csgoId) != blackboard.playerToPath.end()) {
            Path & curPath = blackboard.playerToPath[treeThinker.csgoId];

            if (computeDistance(state.getClient(treeThinker.csgoId).getFootPosForPlayer(), curPath.waypoints[curPath.curWaypoint]) <
                MIN_DISTANCE_TO_NAV_POINT) {
                if (curPath.curWaypoint < curPath.waypoints.size() - 1) {
                    curPath.curWaypoint++;
                }
            }

            playerNodeState[treeThinker.csgoId] = NodeState::Running;
            return playerNodeState[treeThinker.csgoId];
        }

        // otherwise, either no old path or old path is out of date, so update it
        Path newPath;
        auto optionalWaypoints = blackboard.navFile.find_path(vec3Conv(state.getClient(treeThinker.csgoId).getFootPosForPlayer()),
                                                              vec3Conv(curPriority.targetPos));
        if (optionalWaypoints) {
            newPath.pathCallSucceeded = true;
            vector<nav_mesh::vec3_t> tmpWaypoints = optionalWaypoints.value();
            for (const auto & tmpWaypoint : tmpWaypoints) {
                newPath.waypoints.push_back(vec3tConv(tmpWaypoint));
            }
            newPath.curWaypoint = 0;
            newPath.pathEndAreaId =
                    blackboard.navFile.get_nearest_area_by_position(vec3Conv(curPriority.targetPos)).get_id();
        }
        else {
            // do nothing if the pathing call succeeded
            newPath.pathCallSucceeded = false;
        }
        blackboard.playerToPath[treeThinker.csgoId] = newPath;

        playerNodeState[treeThinker.csgoId] = newPath.pathCallSucceeded ? NodeState::Running : NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
    }

    NodeState FireSelectionTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];

        // not executing shooting if no target
        if (curPriority.targetPlayer.playerId == INVALID_ID) {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }

        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        const ServerState::Client & targetClient = state.getClient(curPriority.targetPlayer.playerId);
        double distance = computeDistance(curClient.getFootPosForPlayer(), targetClient.getFootPosForPlayer());

        // if close enough to move and shoot, crouch
        bool shouldCrouch = distance <= treeThinker.engagementParams.standDistance;
        if (distance <= treeThinker.engagementParams.moveDistance) {
            curPriority.movementOptions = {true, false, true};
            curPriority.shootOptions = PriorityShootOptions::Spray;
        }
        else if (distance <= treeThinker.engagementParams.sprayDistance) {
            curPriority.movementOptions = {false, false, shouldCrouch};
            curPriority.shootOptions = PriorityShootOptions::Spray;
        }
        else if (distance <= treeThinker.engagementParams.burstDistance) {
            curPriority.movementOptions = {false, false, shouldCrouch};
            curPriority.shootOptions = PriorityShootOptions::Burst;
        }
        else {
            curPriority.movementOptions = {false, false, shouldCrouch};
            curPriority.shootOptions = PriorityShootOptions::Tap;
        }
        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}

