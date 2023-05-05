//
// Created by durst on 4/13/23.
//

#include "queries/inference_moments/inference_latent_order_helpers.h"
#include "bots/analysis/learned_models.h"

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
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4Pos[rowIndex].x));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4Pos[rowIndex].y));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4Pos[rowIndex].z));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4TicksSincePlant[rowIndex]));
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
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.footPos[rowIndex].x));
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.footPos[rowIndex].y));
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.footPos[rowIndex].z));
                for (size_t priorTick = 0; priorTick < csknow::feature_store::num_prior_ticks; priorTick++) {
                    result.rowCPP.push_back(static_cast<float>(
                                                columnPlayerData.priorFootPos[priorTick][rowIndex].x));
                    result.rowCPP.push_back(static_cast<float>(
                                                columnPlayerData.priorFootPos[priorTick][rowIndex].y));
                    result.rowCPP.push_back(static_cast<float>(
                                                columnPlayerData.priorFootPos[priorTick][rowIndex].z));
                }
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.velocity[rowIndex].x));
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.velocity[rowIndex].y));
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.velocity[rowIndex].z));
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
        ctColumnData = true;
        // distribution cat data
        for (const auto & columnData :
            featureStoreResult.teamFeatureStoreResult.getAllColumnData()) {
            for (size_t playerNum = 0; playerNum < csknow::feature_store::maxEnemies; playerNum++) {
                const auto & columnPlayerData = columnData.get()[playerNum];
                result.playerIdToColumnIndex[columnPlayerData.playerId[rowIndex]] =
                    playerNum + (ctColumnData ? 0 : csknow::feature_store::maxEnemies);
                for (size_t placeIndex = 0; placeIndex < csknow::feature_store::num_places; placeIndex++) {
                    result.rowCPP.push_back(static_cast<float>(
                                                columnPlayerData.curPlace[placeIndex][rowIndex]));
                }
                for (size_t areaIndex = 0; areaIndex < csknow::feature_store::area_grid_size; areaIndex++) {
                    result.rowCPP.push_back(static_cast<float>(
                                                columnPlayerData.areaGridCellInPlace[areaIndex][rowIndex]));
                }
                for (size_t playerIndexOnTeam = 0; playerIndexOnTeam < csknow::feature_store::maxEnemies; playerIndexOnTeam++) {
                    result.rowCPP.push_back(static_cast<float>(
                                                    columnPlayerData.indexOnTeam[playerIndexOnTeam][rowIndex]));
                }
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.ctTeam[rowIndex]));
            }
            ctColumnData = false;
        }
        return result;
    }

    InferenceOrderPlayerAtTickProbabilities extractFeatureStoreOrderResults(
        const at::Tensor & output, const InferenceOrderTickValues & values, int64_t curPlayerId) {
        InferenceOrderPlayerAtTickProbabilities result;
        float mostLikelyOrderProb = -1;
        size_t playerStartIndex = values.playerIdToColumnIndex.at(curPlayerId) * total_orders;
        for (size_t orderIndex = 0; orderIndex < total_orders; orderIndex++) {
            if (useRealProb) {
                result.orderProbabilities.push_back(output[0][playerStartIndex + orderIndex].item<float>());
            }
            else {
                result.orderProbabilities.push_back(1. / total_orders);
            }
            if (result.orderProbabilities.back() > mostLikelyOrderProb) {
                mostLikelyOrderProb = result.orderProbabilities.back();
                result.mostLikelyOrder = static_cast<OrderRole>(orderIndex);
            }
        }
        return result;
    }
}