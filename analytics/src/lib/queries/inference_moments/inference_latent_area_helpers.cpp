//
// Created by durst on 4/20/23.
//

#include "queries/inference_moments/inference_latent_area_helpers.h"

namespace csknow::inference_latent_area {
    InferenceAreaTickValues extractFeatureStoreAreaValues(
            const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex) {
        InferenceAreaTickValues result;
        const csknow::feature_store::TeamFeatureStoreResult & teamFeatureStoreResult =
                featureStoreResult.teamFeatureStoreResult;
        // c4 float data
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4DistanceToASite[rowIndex]));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4DistanceToBSite[rowIndex]));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4Pos[rowIndex].x));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4Pos[rowIndex].y));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4Pos[rowIndex].z));
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
                for (size_t placeIndex = 0; placeIndex < csknow::feature_store::num_places; placeIndex++) {
                    result.rowCPP.push_back(static_cast<float>(
                                                    columnPlayerData.curPlace[placeIndex][rowIndex]));
                }
                for (size_t areaIndex = 0; areaIndex < csknow::feature_store::area_grid_size; areaIndex++) {
                    result.rowCPP.push_back(static_cast<float>(
                                                    columnPlayerData.areaGridCellInPlace[areaIndex][rowIndex]));
                }
            }
            ctColumnData = false;
        }
        // cat data
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4Status[rowIndex]));
        return result;
    }

    InferenceAreaPlayerAtTickProbabilities extractFeatureStoreAreaResults(
            const at::Tensor & output, const InferenceAreaTickValues & values, int64_t curPlayerId) {
        InferenceAreaPlayerAtTickProbabilities result;
        float mostLikelyAreaProb = -1;
        size_t playerStartIndex = values.playerIdToColumnIndex.at(curPlayerId) * csknow::feature_store::area_grid_size;
        for (size_t areaIndex = 0; areaIndex < csknow::feature_store::area_grid_size; areaIndex++) {
            result.areaProbabilities.push_back(output[0][playerStartIndex + areaIndex].item<float>());
            if (result.areaProbabilities.back() > mostLikelyAreaProb) {
                mostLikelyAreaProb = result.areaProbabilities.back();
                result.mostLikelyArea = areaIndex;
            }
        }
        return result;

    }
}
