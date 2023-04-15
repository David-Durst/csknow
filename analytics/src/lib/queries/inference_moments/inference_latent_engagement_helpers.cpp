//
// Created by durst on 4/13/23.
//

#include "queries/inference_moments/inference_latent_engagement_helpers.h"

namespace csknow::inference_latent_engagement {
    InferenceEngagementTickValues extractFeatureStoreEngagementValues(
        const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex) {
        InferenceEngagementTickValues result;
        // seperate different input types
        for (size_t enemyNum = 0; enemyNum < csknow::feature_store::maxEnemies; enemyNum++) {
            const csknow::feature_store::FeatureStoreResult::ColumnEnemyData &columnEnemyData =
                featureStoreResult.columnEnemyData[enemyNum];
            result.enemyIds.push_back(columnEnemyData.playerId[rowIndex]);
            result.rowCPP.push_back(
                static_cast<float>(columnEnemyData.timeSinceLastVisibleOrToBecomeVisible[rowIndex]));
            result.rowCPP.push_back(static_cast<float>(columnEnemyData.worldDistanceToEnemy[rowIndex]));
            result.rowCPP.push_back(static_cast<float>(columnEnemyData.crosshairDistanceToEnemy[rowIndex]));
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

    InferenceEngagementTickProbabilities extractFeatureStoreEngagementResults(
        //const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex, int64_t tickIndex,
        const at::Tensor & output, const InferenceEngagementTickValues & values) {
        InferenceEngagementTickProbabilities result;
        float mostLikelyEnemyProb = -1;
        result.mostLikelyEnemyNum = csknow::feature_store::maxEnemies + 1;
        for (size_t enemyNum = 0; enemyNum <= csknow::feature_store::maxEnemies; enemyNum++) {
            //std::cout << output[0][enemyNum].item<float>() << std::endl;
            result.enemyProbabilities.push_back(output[0][enemyNum].item<float>());
            if (values.enemyStates[enemyNum] != csknow::feature_store::EngagementEnemyState::None &&
                result.enemyProbabilities.back() > mostLikelyEnemyProb) {
                result.mostLikelyEnemyNum = enemyNum;
                mostLikelyEnemyProb = result.enemyProbabilities.back();
                /*
                if (enemyNum < csknow::feature_store::maxEnemies &&
                    featureStoreResult.columnEnemyData[result.mostLikelyEnemyNum].playerId[rowIndex] == INVALID_ID) {
                    std::cout << "invalid noted played tick id " << tickIndex
                              << " enemy num " << enemyNum
                              << " enegagmenet state " << enumAsInt(values.enemyStates[enemyNum])
                              << " size " << featureStoreResult.size
                              << std::endl;
                }
                 */
            }
        }
        return result;
    }
}
