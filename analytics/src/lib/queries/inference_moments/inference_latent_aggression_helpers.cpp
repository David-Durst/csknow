//
// Created by durst on 4/13/23.
//

#include "queries/inference_moments/inference_latent_aggression_helpers.h"

namespace csknow::inference_latent_aggression {
    InferenceAggressionTickValues extractFeatureStoreAggressionValues(
        const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex) {
        InferenceAggressionTickValues result;
        // seperate different input types
        for (size_t enemyNum = 0; enemyNum < csknow::feature_store::maxEnemies; enemyNum++) {
            const csknow::feature_store::FeatureStoreResult::ColumnEnemyData &columnEnemyData =
                featureStoreResult.columnEnemyData[enemyNum];
            result.rowCPP.push_back(
                static_cast<float>(columnEnemyData.timeSinceLastVisibleOrToBecomeVisible[rowIndex]));
            result.rowCPP.push_back(static_cast<float>(columnEnemyData.worldDistanceToEnemy[rowIndex]));
            result.rowCPP.push_back(static_cast<float>(columnEnemyData.crosshairDistanceToEnemy[rowIndex]));
        }
        for (size_t teammateNum = 0; teammateNum < csknow::feature_store::maxEnemies; teammateNum++) {
            const csknow::feature_store::FeatureStoreResult::ColumnTeammateData &columnTeammateData =
                featureStoreResult.columnTeammateData[teammateNum];
            result.rowCPP.push_back(static_cast<float>(columnTeammateData.teammateWorldDistance[rowIndex]));
            result.rowCPP.push_back(static_cast<float>(columnTeammateData.crosshairDistanceToTeammate[rowIndex]));
        }
        for (size_t enemyNum = 0; enemyNum < csknow::feature_store::maxEnemies; enemyNum++) {
            const csknow::feature_store::FeatureStoreResult::ColumnEnemyData &columnEnemyData =
                featureStoreResult.columnEnemyData[enemyNum];
            result.rowCPP.push_back(static_cast<float>(columnEnemyData.enemyEngagementStates[rowIndex]));
            result.enemyStates.push_back(columnEnemyData.enemyEngagementStates[rowIndex]);
        }
        // add last one for no enemy
        result.enemyStates.push_back(csknow::feature_store::EngagementEnemyState::Visible);
        return result;
    }

    InferenceAggressionTickProbabilities extractFeatureStoreAggressionResults(
        const at::Tensor & output, const InferenceAggressionTickValues & values) {
        InferenceAggressionTickProbabilities result;
        float mostLikelyAggressionProb = -1;
        for (size_t aggressionOption = 0; aggressionOption < csknow::feature_store::numNearestEnemyState;
             aggressionOption++) {
            result.aggressionProbabilities.push_back(output[0][aggressionOption].item<float>());
            if (result.aggressionProbabilities.back() > mostLikelyAggressionProb) {
                mostLikelyAggressionProb = result.aggressionProbabilities.back();
                result.mostLikelyAggression = static_cast<feature_store::NearestEnemyState>(aggressionOption);
            }
        }
        return result;
    }
}