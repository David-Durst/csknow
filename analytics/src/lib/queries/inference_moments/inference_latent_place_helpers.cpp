//
// Created by durst on 4/20/23.
//

#include "queries/inference_moments/inference_latent_place_helpers.h"
#include "bots/analysis/learned_models.h"

namespace csknow::inference_latent_place {
    InferencePlaceTickValues extractFeatureStorePlaceValues(
            const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex) {
        InferencePlaceTickValues result;
        const csknow::feature_store::TeamFeatureStoreResult & teamFeatureStoreResult =
                featureStoreResult.teamFeatureStoreResult;
        // c4 float data
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4DistanceToASite[rowIndex]));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4DistanceToBSite[rowIndex]));
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
            }
            ctColumnData = false;
        }
        // cat data
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4Status[rowIndex]));
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
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.ctTeam[rowIndex]));
            }
            ctColumnData = false;
        }
        return result;
    }

    InferencePlacePlayerAtTickProbabilities extractFeatureStorePlaceResults(
            const at::Tensor & output, const InferencePlaceTickValues & values, int64_t curPlayerId) {
        InferencePlacePlayerAtTickProbabilities result;
        float mostLikelyPlaceProb = -1;
        size_t playerStartIndex = values.playerIdToColumnIndex.at(curPlayerId) * csknow::feature_store::num_places;
        for (size_t placeIndex = 0; placeIndex < csknow::feature_store::num_places; placeIndex++) {
            if (useRealProb) {
                result.placeProbabilities.push_back(output[0][playerStartIndex + placeIndex].item<float>());
            }
            else {
                result.placeProbabilities.push_back(1. / csknow::feature_store::num_places);
            }
            if (result.placeProbabilities.back() > mostLikelyPlaceProb) {
                mostLikelyPlaceProb = result.placeProbabilities.back();
                result.mostLikelyPlace = static_cast<PlaceIndex>(placeIndex);
            }
        }
        return result;

    }
}
