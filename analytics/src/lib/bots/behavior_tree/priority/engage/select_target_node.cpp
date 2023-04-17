//
// Created by durst on 5/6/22.
//

#include "bots/behavior_tree/priority/engage_node.h"
#include <functional>

namespace engage {
    constexpr bool useTargetModelProbabilities = true;

    NodeState SelectTargetNode::exec(const ServerState & state, TreeThinker &treeThinker) {
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        vector<std::reference_wrapper<const ServerState::Client>> visibleEnemies =
                state.getVisibleEnemies(treeThinker.csgoId);
        bool havePriority = blackboard.playerToPriority.find(treeThinker.csgoId) != blackboard.playerToPriority.end();
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        TargetPlayer & curTarget = curPriority.targetPlayer;
        curPriority.nonDangerAimArea = {};
        const map<CSGOId, EnemyPositionMemory> & rememberedEnemies = blackboard.playerToMemory[treeThinker.csgoId].positions;
        const map<CSGOId, EnemyPositionMemory> & communicatedEnemies = blackboard.playerToRelevantCommunicatedEnemies[treeThinker.csgoId];

        // if no priority yet or switching from order, setup priority with right type
        if (!havePriority || curPriority.priorityType != PriorityType::Engagement) {
            curPriority.priorityType = PriorityType::Engagement;
        }

        if (!blackboard.inAnalysis && useTargetModelProbabilities) {
            CSGOId targetId = assignPlayerToTargetProbabilistic(curClient, state, curTarget,
                                                                rememberedEnemies, communicatedEnemies);
            if (targetId != INVALID_ID) {
                curPriority.targetPos = curTarget.footPos;
                playerNodeState[treeThinker.csgoId] = NodeState::Success;
            }
            else {
                playerNodeState[treeThinker.csgoId] = NodeState::Failure;
            }
            return playerNodeState[treeThinker.csgoId];
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
                    /*
                    if (curTarget.footPos != oldTargetClient.getFootPosForPlayer()) {
                        int x = 1;
                    }
                    */
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
            int64_t closestI = INVALID_ID;
            double closestDistance = std::numeric_limits<double>::max();
            for (size_t i = 0; i < targetOptions.size(); i++) {
                double newDistance = computeDistance(curClient.getFootPosForPlayer(), targetOptions[i].footPos);
                if (closestI == INVALID_ID || newDistance < closestDistance) {
                    closestDistance = newDistance;
                    closestI = static_cast<int64_t>(i);
                }
            }
            curTarget = targetOptions[closestI];
            // pick random nearby area for target pos
            curPriority.targetPos = curTarget.footPos;
        }

        if (curTarget.playerId == INVALID_ID) {
            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        }
        else {
            playerNodeState[treeThinker.csgoId] = NodeState::Success;
        }
        return playerNodeState[treeThinker.csgoId];
    }

    CSGOId SelectTargetNode::assignPlayerToTargetProbabilistic(
        const ServerState::Client & client, const ServerState & state, TargetPlayer & curTarget,
        const map<CSGOId, EnemyPositionMemory> & rememberedEnemies,
        const map<CSGOId, EnemyPositionMemory> & communicatedEnemies) {
        if (blackboard.inferenceManager.playerToInferenceData.find(client.csgoId) ==
            blackboard.inferenceManager.playerToInferenceData.end() ||
            !blackboard.inferenceManager.playerToInferenceData.at(client.csgoId).validData) {
            return INVALID_ID;
        }
        // select from alive players
        vector<CSGOId> playerIds;
        vector<double> probabilities;
        const csknow::inference_latent_engagement::InferenceEngagementTickValues & engagementTickValues =
            blackboard.inferenceManager.playerToInferenceData.at(client.csgoId).engagementValues;
        const csknow::inference_latent_engagement::InferenceEngagementTickProbabilities & engagementProbabilities =
            blackboard.inferenceManager.playerToInferenceData.at(client.csgoId).engagementProbabilities;
        //size_t printedTimes = 0;
        //std::cout << "checking target for " << client.csgoId << std::endl;
        for (size_t i = 0; i < engagementTickValues.enemyIds.size(); i++) {
            if (engagementTickValues.enemyIds[i] != INVALID_ID) {
                playerIds.push_back(engagementTickValues.enemyIds[i]);
                probabilities.push_back(engagementProbabilities.enemyProbabilities[i]);
                //std::cout << "enemy id " << playerIds.back() << " raw prob " << probabilities.back() << std::endl;
                //printedTimes++;
            }
        }
        /*
        if (probabilities.empty() && !state.getVisibleEnemies(client.csgoId).empty()) {
            std::cout << "not empty mismatch" << std::endl;
        }
        if (probabilities.empty()) {
            for (size_t i = 0; i < engagementTickValues.enemyIds.size(); i++) {
                if (engagementTickValues.enemyStates[i] != csknow::feature_store::EngagementEnemyState::None) {
                    std::cout << "no enemy id but valid engagemnet state" << std::endl;
                }
            }
        }
         */

        // re-weight just for one site
        double reweightFactor = 0.;
        for (size_t i = 0; i < probabilities.size(); i++) {
            reweightFactor += probabilities[i];
        }
        for (size_t i = 0; i < probabilities.size(); i++) {
            probabilities[i] *= 1/reweightFactor;
        }
        double probSample = blackboard.aggressionDis(blackboard.gen);
        double weightSoFar = 0.;
        CSGOId targetId = INVALID_ID;
        for (size_t i = 0; i < probabilities.size(); i++) {
            weightSoFar += probabilities[i];
            /*
            if (client.csgoId == 6) {
                std::cout << "playerIds " << i << " for real id " << playerIds[i] << " has prob " << probabilities[i] << " compared to " << weightSoFar << std::endl;
            }
             */
            if (probSample < weightSoFar) {
                //std::cout << "assigning to " << client.team << ", " << i << std::endl;
                targetId = playerIds[i];
                break;
            }
        }
        /*
        if (client.csgoId == 6) {
            std::cout << "selected target id " << targetId << std::endl;
            std::cout << "communicated enemies size " << communicatedEnemies.size() << std::endl;
            for (const auto & [communicatedEnemyId, _] : communicatedEnemies) {
                std::cout << "communicated enemy id " << communicatedEnemyId << std::endl;
            }
        }
         */
        // default if probs don't sum perfectly is take last one as this will result from a
        // slight numerical instability mismatch

        if (targetId != INVALID_ID) {
            const ServerState::Client & curTargetClient = state.getClient(targetId);
            if (state.isVisible(client.csgoId, targetId)) {
                curTarget = {curTargetClient.csgoId, state.roundNumber, client.lastFrame,
                             curTargetClient.getFootPosForPlayer(), curTargetClient.getEyePosForPlayer(), true};
            }
            else if (rememberedEnemies.find(targetId) != rememberedEnemies.end()) {
                curTarget = {curTargetClient.csgoId, state.roundNumber, client.lastFrame,
                             rememberedEnemies.find(curTarget.playerId)->second.lastSeenFootPos,
                             rememberedEnemies.find(curTarget.playerId)->second.lastSeenEyePos,
                             false};
            }
            else if (communicatedEnemies.find(targetId) != communicatedEnemies.end()) {
                /*
                if (client.csgoId == 6) {
                    std::cout << "should be handling communicated enemy " << std::endl;
                }
                 */
                curTarget = {curTargetClient.csgoId, state.roundNumber, client.lastFrame,
                    communicatedEnemies.find(curTarget.playerId)->second.lastSeenFootPos,
                    communicatedEnemies.find(curTarget.playerId)->second.lastSeenEyePos,
                    false};
            }
        }
        else {
            // don't worry about this case
            // it's because of late updates so enemies become visible before model reruns
            /*
            std::cout << "bad target assignment due to overflow" << std::endl;
            std::cout << "id, prob ";
            for (size_t i = 0; i < probabilities.size(); i++) {
                std::cout << playerIds[i] << "," << probabilities[i] << " ; ";
            }
            std::cout << std::endl;
             */
        }
        /*
        if (client.csgoId == 6) {
            std::cout << "cur id " << client.csgoId << " enemy id " << curTarget.playerId << std::endl;
        }
         */
        return targetId;
    }
}

