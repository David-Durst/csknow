//
// Created by durst on 4/15/23.
//

#include "bots/behavior_tree/priority/engage_node.h"

namespace engage {
    constexpr bool useAggressionModelProbabilities = true;

    NodeState SelectTargetAggressionNode::exec(const ServerState &state, TreeThinker &treeThinker) {
        const OrderId & curOrderId = blackboard.strategy.getOrderIdForPlayer(treeThinker.csgoId);
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

            if (aggressionOption == enumAsInt(csknow::feature_store::NearestEnemyState::Decrease)) {
                curPriority.moveOptions = {true, false, true};
            }
            else if (aggressionOption == enumAsInt(csknow::feature_store::NearestEnemyState::Constant)) {
                curPriority.moveOptions = {false, false, true};
            }
            else {
                curPriority.moveOptions = {true, false, true};
            }

            playerNodeState[treeThinker.csgoId] = NodeState::Failure;
            return playerNodeState[treeThinker.csgoId];
        }

        playerNodeState[treeThinker.csgoId] = NodeState::Success;
        return playerNodeState[treeThinker.csgoId];
    }
}
