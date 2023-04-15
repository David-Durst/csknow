//
// Created by durst on 4/15/23.
//

#include "bots/behavior_tree/priority/engage_node.h"

namespace engage {
    constexpr bool useAggressionModelProbabilities = false;

    NodeState SelectTargetAggressionNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        Priority & curPriority = blackboard.playerToPriority[treeThinker.csgoId];
        //Path & curPath = blackboard.playerToPath[treeThinker.csgoId];

        if (!blackboard.inTest && useAggressionModelProbabilities) {
            vector<double> probabilities;
            const csknow::inference_latent_aggression::InferenceAggressionTickProbabilities & aggressionProbabilities =
                blackboard.inferenceManager.playerToInferenceData.at(treeThinker.csgoId).aggressionProbabilities;

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

            if (aggressionOption == static_cast<size_t>(csknow::feature_store::NearestEnemyState::Decrease)) {
                curPriority.moveOptions = {true, false, true};
                curPriority.targetPos = curPriority.targetPlayer.footPos;
            }
            else if (aggressionOption == static_cast<size_t>(csknow::feature_store::NearestEnemyState::Constant)) {
                curPriority.moveOptions = {false, false, true};
            }
            else {
                // move to nearest area not visible to enemy
                curPriority.moveOptions = {true, false, true};
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
                            curAreaDistance = minAreaDistance;
                            minAreaIndex = areaIndex;
                        }
                    }
                }
                curPriority.targetPos = blackboard.visPoints.getCellVisPoints()[minAreaIndex].center;
            }

            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
