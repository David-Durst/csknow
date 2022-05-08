//
// Created by durst on 5/6/22.
//

#include "bots/behavior_tree/priority/engage_node.h"
#include <functional>

namespace engage {
    NodeState TargetSelectionTaskNode::exec(const ServerState & state, TreeThinker &treeThinker) {
        // keep same target if from this round and still visible
        if (blackboard.playerToTarget.find(treeThinker.csgoId) != blackboard.playerToTarget.end()) {
            TargetPlayer & oldTarget = blackboard.playerToTarget[treeThinker.csgoId];
            if (oldTarget.round == state.roundNumber) {
                const ServerState::Client & oldTargetClient =
                        state.clients[state.csgoIdToCSKnowId[oldTarget.targetPlayer]];
                if (oldTargetClient.isAlive && state.isVisible(treeThinker.csgoId, oldTarget.targetPlayer)) {
                    oldTarget.firstTargetFrame++;
                    nodeState = NodeState::Success;
                    return nodeState;
                }
            }
        }

        // find all visible, alive enemies
        const ServerState::Client curClient = state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]];
        vector<std::reference_wrapper<const ServerState::Client>> visibleEnemies;
        for (const auto & otherClient : state.clients) {
            if (otherClient.team != curClient.team && otherClient.isAlive &&
                state.isVisible(treeThinker.csgoId, otherClient.csgoId)) {
                std::reference_wrapper<const ServerState::Client> b = otherClient;
                visibleEnemies.push_back(otherClient);
            }
        }

        // remove from targets list if no target as none visible
        if (visibleEnemies.empty()) {
            blackboard.playerToTarget.erase(treeThinker.csgoId);
        }
        // otherwise, assign to nearest enemy
        else {
            int64_t closestId = INVALID_ID;
            double closestDistance = std::numeric_limits<double>::max();
            for (size_t i = 0; i < visibleEnemies.size(); i++) {
                double newDistance = computeDistance(curClient.getFootPosForPlayer(), visibleEnemies[i].get().getFootPosForPlayer());
                if (closestId == INVALID_ID || newDistance < closestDistance) {
                    closestDistance = newDistance;
                    closestId = visibleEnemies[i].get().csgoId;
                }
            }
            blackboard.playerToTarget[treeThinker.csgoId] = {closestId, state.roundNumber, curClient.lastFrame};
        }

        nodeState = NodeState::Success;
        return nodeState;
    }


    NodeState ShootTaskNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        // not executing shooting if no target
        if (blackboard.playerToTarget.find(treeThinker.csgoId) == blackboard.playerToTarget.end()) {
            nodeState = NodeState::Failure;
            return nodeState;
        }

        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        const ServerState::Client & curClient = state.clients[state.csgoIdToCSKnowId[treeThinker.csgoId]];
        const ServerState::Client & targetClient =
                state.clients[state.csgoIdToCSKnowId[blackboard.playerToTarget[treeThinker.csgoId].targetPlayer]];
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
        nodeState = NodeState::Success;
        return nodeState;
    }
}