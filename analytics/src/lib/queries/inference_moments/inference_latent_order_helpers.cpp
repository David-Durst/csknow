//
// Created by durst on 4/13/23.
//

#include "queries/inference_moments/inference_latent_order_helpers.h"

namespace csknow::inference_latent_order {
    InferenceOrderTickValues extractFeatureStoreOrderValues(
        const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex) {
        InferenceOrderTickValues result;
        const csknow::feature_store::TeamFeatureStoreResult & teamFeatureStoreResult =
            featureStoreResult.teamFeatureStoreResult;
        // c4 float data
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4DistanceToASite[rowIndex]));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4DistanceToBSite[rowIndex]));
        for (size_t orderIndex = 0; orderIndex < csknow::feature_store::num_orders_per_site; orderIndex++) {
            result.rowCPP.push_back(static_cast<float>(
                                 teamFeatureStoreResult.c4DistanceToNearestAOrderNavArea[orderIndex][rowIndex]));
        }
        for (size_t orderIndex = 0; orderIndex < csknow::feature_store::num_orders_per_site; orderIndex++) {
            result.rowCPP.push_back(static_cast<float>(
                                 teamFeatureStoreResult.c4DistanceToNearestBOrderNavArea[orderIndex][rowIndex]));
        }
        // player data
        bool ctColumnData = true;
        for (const auto & columnData :
            featureStoreResult.teamFeatureStoreResult.getAllColumnData()) {
            for (size_t playerNum = 0; playerNum < csknow::feature_store::maxEnemies; playerNum++) {
                const auto & columnPlayerData = columnData.get()[playerNum];
                result.playerIdToColumnIndex[columnPlayerData.playerId[rowIndex]] =
                    playerNum + (ctColumnData ? 0 : csknow::feature_store::maxEnemies);
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.distanceToASite[rowIndex]));
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.distanceToBSite[rowIndex]));
                for (size_t orderIndex = 0; orderIndex < csknow::feature_store::num_orders_per_site; orderIndex++) {
                    result.rowCPP.push_back(static_cast<float>(
                                         columnPlayerData.distanceToNearestAOrderNavArea[orderIndex][rowIndex]));
                }
                for (size_t orderIndex = 0; orderIndex < csknow::feature_store::num_orders_per_site; orderIndex++) {
                    result.rowCPP.push_back(static_cast<float>(
                                         columnPlayerData.distanceToNearestBOrderNavArea[orderIndex][rowIndex]));
                }
            }
            ctColumnData = false;
        }
        // cat data
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4Status[rowIndex]));
        return result;
    }

    InferenceOrderPlayerAtTickProbabilities extractFeatureStoreOrderResults(
        const at::Tensor & output, const InferenceOrderTickValues & values, int64_t curPlayerId) {
        InferenceOrderPlayerAtTickProbabilities result;
        float mostLikelyOrderProb = -1;
        size_t playerStartIndex = values.playerIdToColumnIndex.at(curPlayerId) * total_orders;
        for (size_t orderIndex = 0; orderIndex < total_orders; orderIndex++) {
            result.orderProbabilities.push_back(output[0][playerStartIndex + orderIndex].item<float>());
            if (result.orderProbabilities.back() > mostLikelyOrderProb) {
                mostLikelyOrderProb = result.orderProbabilities.back();
                result.mostLikelyOrder = static_cast<OrderRole>(orderIndex);
            }
        }
        return result;
    }
}