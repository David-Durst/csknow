//
// Created by durst on 5/6/22.
//

#include "bots/behavior_tree/priority/target_selection_node.h"
#include <functional>

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
    const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
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

