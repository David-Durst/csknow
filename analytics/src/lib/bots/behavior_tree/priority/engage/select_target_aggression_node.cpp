//
// Created by durst on 4/15/23.
//

#include "bots/behavior_tree/priority/engage_node.h"
#include "bots/analysis/learned_models.h"

namespace engage {
    NodeState SelectTargetAggressionNode::exec(const ServerState &, TreeThinker &treeThinker) {
        playerNodeState[treeThinker.csgoId] = NodeState::Failure;
        return playerNodeState[treeThinker.csgoId];
        /*
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        const ServerState::Client & curClient = state.getClient(treeThinker.csgoId);
        //Path & curPath = blackboard.playerToPath[treeThinker.csgoId];

        // forcing targetPos to be something valid
        TargetPlayer & curTarget = curPriority.targetPlayer;
        //curPriority.targetPos = curTarget.footPos;
        setTargetPosForTarget(curClient, curPriority, curTarget);

        if (blackboard.playerToTicksSinceLastProbAggressionAssignment.find(treeThinker.csgoId) ==
            blackboard.playerToTicksSinceLastProbAggressionAssignment.end()) {
            blackboard.playerToTicksSinceLastProbAggressionAssignment[treeThinker.csgoId] = newAggressionTicks;
        }
        blackboard.playerToTicksSinceLastProbAggressionAssignment[treeThinker.csgoId]++;
        bool timeForNewAggression =
                blackboard.playerToTicksSinceLastProbAggressionAssignment.at(treeThinker.csgoId) >= newTargetTicks;
        if (!blackboard.inAnalysis && !blackboard.inTest && getAggressionModelProbabilities(curClient.team) &&
            blackboard.inferenceManager.playerToInferenceData.find(treeThinker.csgoId) !=
            blackboard.inferenceManager.playerToInferenceData.end() &&
            blackboard.inferenceManager.playerToInferenceData.at(treeThinker.csgoId).validData) {
            if (blackboard.playerToLastProbAggressionAssignment.find(treeThinker.csgoId) ==
                blackboard.playerToLastProbAggressionAssignment.end() || timeForNewAggression) {
                blackboard.playerToLastProbAggressionAssignment[treeThinker.csgoId] =
                    csknow::feature_store::NearestEnemyState::NUM_NEAREST_ENEMY_STATE;
            }
            csknow::feature_store::NearestEnemyState & lastProbAggressionAssignment = blackboard.playerToLastProbAggressionAssignment[treeThinker.csgoId];
            const csknow::inference_latent_aggression::InferenceAggressionTickProbabilities & aggressionProbabilities =
                blackboard.inferenceManager.playerToInferenceData.at(treeThinker.csgoId).aggressionProbabilities;
            vector<float> probabilities = aggressionProbabilities.aggressionProbabilities;

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
            size_t aggressionOption = 0;
            for (size_t i = 0; i < probabilities.size(); i++) {
                weightSoFar += probabilities[i];
                if (probSample < weightSoFar) {
                    aggressionOption = i;
                    break;
                }
            }
            // use prior aggression if an option
            if (lastProbAggressionAssignment != csknow::feature_store::NearestEnemyState::NUM_NEAREST_ENEMY_STATE) {
                aggressionOption = static_cast<size_t>(lastProbAggressionAssignment);
            }
            else {
                lastProbAggressionAssignment = static_cast<csknow::feature_store::NearestEnemyState>(aggressionOption);
                blackboard.playerToTicksSinceLastProbAggressionAssignment[treeThinker.csgoId] = 0;
            }

            if (aggressionOption == static_cast<size_t>(csknow::feature_store::NearestEnemyState::Decrease)) {
                curPriority.moveOptions = {true, false, false};
                //curPriority.targetPos = curPriority.targetPlayer.footPos;
            }
            else if (aggressionOption == static_cast<size_t>(csknow::feature_store::NearestEnemyState::Constant)) {
                if (curTarget.visible) {
                    curPriority.moveOptions = {false, false, false};
                }
                else {
                    curPriority.moveOptions = {true, false, false};
                }
            }
            else {
                // move to nearest area not visible to enemy
                curPriority.moveOptions = {true, false, false};
                AreaBits targetVisBits =
                    blackboard.getVisibleAreasByPlayer(state.getClient(curPriority.targetPlayer.playerId));
                const nav_mesh::nav_area & curArea = blackboard.getPlayerNavArea(state.getClient(treeThinker.csgoId));
                size_t curAreaIndex = blackboard.visPoints.areaIdToIndex(curArea.get_id());
                size_t minAreaIndex = 0;
                double minAreaDistance = std::numeric_limits<double>::max();
                for (size_t areaIndex = 0; areaIndex < targetVisBits.size(); areaIndex++) {
                    if (!targetVisBits[areaIndex]) {
                        double curAreaDistance = blackboard.reachability.getDistance(curAreaIndex, areaIndex);
                        if (curAreaDistance < minAreaDistance) {
                            minAreaDistance = curAreaDistance;
                            minAreaIndex = areaIndex;
                        }
                    }
                }
                curPriority.targetPos = blackboard.visPoints.getCellVisPoints()[minAreaIndex].center;
            }
        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
        */
    }


    void SelectTargetAggressionNode::setTargetPosForTarget(const ServerState::Client & client, Priority & curPriority,
                                                           const TargetPlayer & curTarget) {
        AreaId curAreaId = blackboard.navFile
            .get_nearest_area_by_position(vec3Conv(client.getFootPosForPlayer())).get_id();
        size_t curAreaIndex = blackboard.navFile.m_area_ids_to_indices.at(curAreaId);
        AreaId targetAreaId = blackboard.navFile.get_nearest_area_by_position(vec3Conv(curTarget.footPos)).get_id();
        AreaBits enemyDangerAreaBits = blackboard.visPoints.getDangerRelativeToSrc(targetAreaId);
        double nearestDangerAreaDistance = std::numeric_limits<double>::max();
        size_t nearestDangerAreaIndex = 0;
        for (size_t dangerAreaIndex = 0; dangerAreaIndex < enemyDangerAreaBits.size(); dangerAreaIndex++) {
            if (enemyDangerAreaBits[dangerAreaIndex]) {
                double newDangerAreaDistance = blackboard.reachability.getDistance(curAreaIndex, dangerAreaIndex);
                if (newDangerAreaDistance < nearestDangerAreaDistance) {
                    nearestDangerAreaDistance = newDangerAreaDistance;
                    nearestDangerAreaIndex = dangerAreaIndex;
                }
            }
        }
        curPriority.targetPos = getCenter(blackboard.reachability.coordinate[nearestDangerAreaIndex]);
    }
}
