//
// Created by durst on 4/13/23.
//

#include "queries/inference_moments/inference_latent_aggression_helpers.h"
#include "bots/analysis/learned_models.h"
#include "feature_store_precommit.h"

namespace csknow::inference_latent_aggression {
    InferenceAggressionTickValues extractFeatureStoreAggressionValues(
        const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex) {
        InferenceAggressionTickValues result;
        // seperate different input types
        for (size_t enemyNum = 0; enemyNum < feature_store::max_enemies; enemyNum++) {
            const csknow::feature_store::FeatureStoreResult::ColumnEnemyData &columnEnemyData =
                featureStoreResult.columnEnemyData[enemyNum];
            result.rowCPP.push_back(
                static_cast<float>(columnEnemyData.timeSinceLastVisibleOrToBecomeVisible[rowIndex]));
            result.rowCPP.push_back(static_cast<float>(columnEnemyData.worldDistanceToEnemy[rowIndex]));
            result.rowCPP.push_back(static_cast<float>(columnEnemyData.crosshairDistanceToEnemy[rowIndex]));
        }
        for (size_t teammateNum = 0; teammateNum < feature_store::max_enemies; teammateNum++) {
            const csknow::feature_store::FeatureStoreResult::ColumnTeammateData &columnTeammateData =
                featureStoreResult.columnTeammateData[teammateNum];
            result.rowCPP.push_back(static_cast<float>(columnTeammateData.teammateWorldDistance[rowIndex]));
            result.rowCPP.push_back(static_cast<float>(columnTeammateData.crosshairDistanceToTeammate[rowIndex]));
        }
        for (size_t enemyNum = 0; enemyNum < feature_store::max_enemies; enemyNum++) {
            const csknow::feature_store::FeatureStoreResult::ColumnEnemyData &columnEnemyData =
                featureStoreResult.columnEnemyData[enemyNum];
            result.rowCPP.push_back(static_cast<float>(columnEnemyData.enemyEngagementStates[rowIndex]));
            result.enemyStates.push_back(columnEnemyData.enemyEngagementStates[rowIndex]);
        }
        // add last one for no enemy
        result.enemyStates.push_back(csknow::feature_store::EngagementEnemyState::Visible);
        return result;
    }

    InferenceAggressionTickProbabilities extractFeatureStoreAggressionResults(const at::Tensor & output) {
        InferenceAggressionTickProbabilities result;
        float mostLikelyAggressionProb = -1;
        for (size_t aggressionOption = 0; aggressionOption < csknow::feature_store::numNearestEnemyState;
             aggressionOption++) {
            if (false) {
                result.aggressionProbabilities.push_back(output[0][aggressionOption].item<float>());
            }
            else {
                result.aggressionProbabilities.push_back(1. / csknow::feature_store::numNearestEnemyState);
            }
            if (result.aggressionProbabilities.back() > mostLikelyAggressionProb) {
                mostLikelyAggressionProb = result.aggressionProbabilities.back();
                result.mostLikelyAggression = static_cast<feature_store::NearestEnemyState>(aggressionOption);
            }
        }
        return result;
    }
}