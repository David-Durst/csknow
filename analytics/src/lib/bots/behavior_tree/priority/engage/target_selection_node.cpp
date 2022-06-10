//
// Created by durst on 5/6/22.
//

#include "bots/behavior_tree/priority/engage_node.h"
#include <functional>

namespace engage {
    NodeState SelectTargetNode::exec(const ServerState & state, TreeThinker &treeThinker) {
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        TargetPlayer & curTarget = curPriority.targetPlayer;
        // keep same target if from this round and still visible
        if (curTarget.playerId != INVALID_ID) {
            if (curTarget.round == state.roundNumber) {
                const ServerState::Client & oldTargetClient =
                        state.clients[state.csgoIdToCSKnowId[curTarget.playerId]];
                if (oldTargetClient.isAlive && state.isVisible(treeThinker.csgoId, curTarget.playerId)) {
                    playerNodeState[treeThinker.csgoId] = NodeState::Success;
                    return playerNodeState[treeThinker.csgoId];
                }
            }
        }

        // find all visible, alive enemies
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        vector<std::reference_wrapper<const ServerState::Client>> visibleEnemies;
        for (const auto & otherClient : state.clients) {
            if (otherClient.team != curClient.team && otherClient.isAlive &&
                state.isVisible(treeThinker.csgoId, otherClient.csgoId)) {
                visibleEnemies.push_back(otherClient);
            }
        }

        // remove from targets list if no target as none visible
        if (visibleEnemies.empty()) {
            curTarget.playerId = INVALID_ID;
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
            curTarget = {closestId, state.roundNumber, curClient.lastFrame};
        }

        if (curTarget.playerId == INVALID_ID) {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
        }
        return playerNodeState[treeThinker.csgoId];
    }

}

