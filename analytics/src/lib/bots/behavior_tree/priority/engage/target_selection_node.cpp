//
// Created by durst on 5/6/22.
//

#include "bots/behavior_tree/priority/engage_node.h"
#include <functional>

namespace engage {
    NodeState SelectTargetNode::exec(const ServerState & state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        vector<std::reference_wrapper<const ServerState::Client>> visibleEnemies =
                state.getVisibleEnemies(treeThinker.csgoId);
        bool havePriority = blackboard.playerToPriority.find(treeThinker.csgoId) != blackboard.playerToPriority.end();
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        TargetPlayer & curTarget = curPriority.targetPlayer;
        const map<CSGOId, EnemyPositionMemory> & rememberedEnemies = blackboard.playerToMemory[treeThinker.csgoId].positions;
        const map<CSGOId, EnemyPositionMemory> & communicatedEnemies = blackboard.getCommunicatedPlayers(state, treeThinker).positions;

        // if no priority yet or switching from order, setup priority with right type
        if (!havePriority || curPriority.priorityType != PriorityType::Engagement) {
            curPriority.priorityType = PriorityType::Engagement;
        }

        // keep same target if from this round and still visible or remembered or communicated
        // forget about remembered/communicated if another enemy visible
        if (curTarget.playerId != INVALID_ID && curTarget.round == state.roundNumber) {
            const ServerState::Client & oldTargetClient =
                    state.clients[state.csgoIdToCSKnowId[curTarget.playerId]];
            if (oldTargetClient.isAlive) {
                bool continueSameTarget = false;
                if (curTarget.visible && state.isVisible(treeThinker.csgoId, curTarget.playerId)) {
                    curTarget.footPos = oldTargetClient.getFootPosForPlayer();
                    curTarget.eyePos = oldTargetClient.getEyePosForPlayer();
                    continueSameTarget = true;
                }
                else if (!curTarget.visible && visibleEnemies.empty() &&
                        (rememberedEnemies.find(curTarget.playerId) != rememberedEnemies.end())) {
                    curTarget.footPos = rememberedEnemies.find(curTarget.playerId)->second.lastSeenFootPos;
                    if (curTarget.footPos != oldTargetClient.getFootPosForPlayer()) {
                        int x = 1;
                    }
                    curTarget.eyePos = rememberedEnemies.find(curTarget.playerId)->second.lastSeenEyePos;
                    continueSameTarget = true;
                }
                else if (!curTarget.visible && visibleEnemies.empty() &&
                         (communicatedEnemies.find(curTarget.playerId) != communicatedEnemies.end())) {
                    curTarget.footPos = communicatedEnemies.find(curTarget.playerId)->second.lastSeenFootPos;
                    curTarget.eyePos = communicatedEnemies.find(curTarget.playerId)->second.lastSeenEyePos;
                    continueSameTarget = true;
                }
                if (continueSameTarget) {
                    playerNodeState[treeThinker.csgoId] = NodeState::Success;
                    return playerNodeState[treeThinker.csgoId];
                }
            }
        }

        // find all visible, alive enemies
        // if no visible enemies, check if any remembered or communicated enemies
        //vector<std::reference_wrapper<const ServerState::Client>> targetOptions = visibleEnemies;
        vector<TargetPlayer> targetOptions;
        for (const auto visibleEnemy : visibleEnemies) {
            targetOptions.push_back({visibleEnemy.get().csgoId, state.roundNumber, curClient.lastFrame,
                                     visibleEnemy.get().getFootPosForPlayer(), visibleEnemy.get().getEyePosForPlayer(), true});
        }
        if (targetOptions.empty()) {
            for (const auto [id, enemyPositionMemory] : rememberedEnemies) {
                targetOptions.push_back({id, state.roundNumber, curClient.lastFrame,
                                         enemyPositionMemory.lastSeenFootPos, enemyPositionMemory.lastSeenEyePos, false});
            }
            for (const auto [id, enemyPositionMemory] : communicatedEnemies) {
                targetOptions.push_back({id, state.roundNumber, curClient.lastFrame,
                                         enemyPositionMemory.lastSeenFootPos, enemyPositionMemory.lastSeenEyePos, false});
            }
        }

        // remove from targets list if no target as none visible
        if (targetOptions.empty()) {
            curTarget.playerId = INVALID_ID;
        }
        // otherwise, assign to nearest enemy
        else {
            size_t closestI = INVALID_ID;
            double closestDistance = std::numeric_limits<double>::max();
            for (size_t i = 0; i < targetOptions.size(); i++) {
                double newDistance = computeDistance(curClient.getFootPosForPlayer(), targetOptions[i].footPos);
                if (closestI == INVALID_ID || newDistance < closestDistance) {
                    closestDistance = newDistance;
                    closestI = i;
                }
            }
            curTarget = targetOptions[closestI];
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

