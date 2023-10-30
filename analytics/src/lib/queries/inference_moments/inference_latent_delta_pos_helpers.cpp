//
// Created by durst on 4/20/23.
//

#include "queries/inference_moments/inference_latent_delta_pos_helpers.h"
#include "bots/analysis/learned_models.h"

namespace csknow::inference_delta_pos {
    InferenceDeltaPosTickValues extractFeatureStoreDeltaPosValues(
            const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex,
            TeamSaveControlParameters teamSaveControlParameters) {
        InferenceDeltaPosTickValues result;
        const csknow::feature_store::TeamFeatureStoreResult & teamFeatureStoreResult =
                featureStoreResult.teamFeatureStoreResult;
        // c4 float data
        /*
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4DistanceToASite[rowIndex]));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4DistanceToBSite[rowIndex]));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4Pos[rowIndex].x));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4Pos[rowIndex].y));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4Pos[rowIndex].z));
         */
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4TimeLeftPercent[rowIndex]));

        // player data
        bool ctColumnData = true;
        for (const auto & columnData :
                featureStoreResult.teamFeatureStoreResult.getAllColumnData()) {
            for (size_t playerNum = 0; playerNum < feature_store::max_enemies; playerNum++) {
                const auto & columnPlayerData = columnData.get()[playerNum];
                result.playerIdToColumnIndex[columnPlayerData.playerId[rowIndex]] =
                        playerNum + (ctColumnData ? 0 : feature_store::max_enemies);
                /*
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.alignedFootPos[rowIndex].x));
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.alignedFootPos[rowIndex].y));
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.alignedFootPos[rowIndex].z));
                */
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.footPos[rowIndex].x));
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.footPos[rowIndex].y));
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.footPos[rowIndex].z));
                for (size_t priorTick = 0; priorTick < csknow::feature_store::num_prior_ticks_inference; priorTick++) {
                    result.rowCPP.push_back(static_cast<float>(
                                                columnPlayerData.priorFootPos[priorTick][rowIndex].x));
                    result.rowCPP.push_back(static_cast<float>(
                                                columnPlayerData.priorFootPos[priorTick][rowIndex].y));
                    result.rowCPP.push_back(static_cast<float>(
                                                columnPlayerData.priorFootPos[priorTick][rowIndex].z));
                }
                result.rowCPP.push_back(columnPlayerData.nearestCrosshairDistanceToEnemy[rowIndex]);
                for (size_t priorTick = 0; priorTick < csknow::feature_store::num_prior_ticks_inference; priorTick++) {
                    result.rowCPP.push_back(columnPlayerData.priorNearestCrosshairDistanceToEnemy[priorTick][rowIndex]);
                }
                result.rowCPP.push_back(columnPlayerData.hurtInLast5s[rowIndex]);
                result.rowCPP.push_back(columnPlayerData.fireInLast5s[rowIndex]);
                result.rowCPP.push_back(columnPlayerData.noFOVEnemyVisibleInLast5s[rowIndex]);
                result.rowCPP.push_back(columnPlayerData.fovEnemyVisibleInLast5s[rowIndex]);
                result.rowCPP.push_back(columnPlayerData.health[rowIndex]);
                result.rowCPP.push_back(columnPlayerData.armor[rowIndex]);
                result.rowCPP.push_back(columnPlayerData.secondsAfterPriorHitEnemy[rowIndex]);
                result.rowCPP.push_back(columnPlayerData.secondsUntilNextHitEnemy[rowIndex]);
                /*
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.velocity[rowIndex].x));
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.velocity[rowIndex].y));
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.velocity[rowIndex].z));
                 */
            }
            ctColumnData = false;
        }
        // distribution cat data
        ctColumnData = true;
        for (const auto & columnData :
            featureStoreResult.teamFeatureStoreResult.getAllColumnData()) {
            for (size_t playerNum = 0; playerNum < feature_store::max_enemies; playerNum++) {
                const auto & columnPlayerData = columnData.get()[playerNum];
                result.playerIdToColumnIndex[columnPlayerData.playerId[rowIndex]] =
                    playerNum + (ctColumnData ? 0 : feature_store::max_enemies);
                /*
                for (size_t placeIndex = 0; placeIndex < csknow::feature_store::num_places; placeIndex++) {
                    result.rowCPP.push_back(static_cast<float>(
                                                    columnPlayerData.curPlace[placeIndex][rowIndex]));
                }
                for (size_t areaIndex = 0; areaIndex < csknow::feature_store::area_grid_size; areaIndex++) {
                    result.rowCPP.push_back(static_cast<float>(
                                                    columnPlayerData.areaGridCellInPlace[areaIndex][rowIndex]));
                }
                 */
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.alive[rowIndex]));
                result.rowCPP.push_back(static_cast<float>(columnPlayerData.ctTeam[rowIndex]));
                if (columnPlayerData.alive[rowIndex]) {
                    result.rowCPP.push_back(
                            teamSaveControlParameters.getPushModelValue(false, feature_store::DecreaseTimingOption::s5,
                                                                        columnPlayerData.playerId[rowIndex]));
                    result.rowCPP.push_back(
                            teamSaveControlParameters.getPushModelValue(false, feature_store::DecreaseTimingOption::s10,
                                                                        columnPlayerData.playerId[rowIndex]));
                    result.rowCPP.push_back(
                            teamSaveControlParameters.getPushModelValue(false, feature_store::DecreaseTimingOption::s20,
                                                                        columnPlayerData.playerId[rowIndex]));
                }
                else {
                    result.rowCPP.push_back(0.f);
                    result.rowCPP.push_back(0.f);
                    result.rowCPP.push_back(0.f);
                }
            }
            ctColumnData = false;
        }
        //result.rowCPP.push_back(teamFeatureStoreResult.curBaiting ? 1.0f : 0.f);
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4PlantA[rowIndex]));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4PlantB[rowIndex]));
        result.rowCPP.push_back(static_cast<float>(teamFeatureStoreResult.c4NotPlanted[rowIndex]));
        return result;
    }

    InferenceDeltaPosPlayerAtTickProbabilities extractFeatureStoreDeltaPosResults(
            const at::Tensor & output, const InferenceDeltaPosTickValues & values, int64_t curPlayerId, TeamId teamId) {
        InferenceDeltaPosPlayerAtTickProbabilities result;
        /*
        float mostLikelyDeltaPosProb = -1;
        size_t playerStartIndex = values.playerIdToColumnIndex.at(curPlayerId) * csknow::feature_store::delta_pos_grid_num_cells;
        for (size_t deltaPosIndex = 0; deltaPosIndex < csknow::feature_store::delta_pos_grid_num_cells; deltaPosIndex++) {
            if ((useRealProbT && teamId == ENGINE_TEAM_T) || (useRealProbCT && teamId == ENGINE_TEAM_CT)) {
                result.deltaPosProbabilities.push_back(output[0][playerStartIndex + deltaPosIndex].item<float>());
            }
            else {
                result.deltaPosProbabilities.push_back(1. / csknow::feature_store::delta_pos_grid_num_cells);
            }
            if (result.deltaPosProbabilities.back() > mostLikelyDeltaPosProb) {
                mostLikelyDeltaPosProb = result.deltaPosProbabilities.back();
                result.mostLikelyDeltaPos = static_cast<int64_t>(deltaPosIndex);
            }
        }
        */

        float mostLikelyRadialVelProb = -1;
        size_t columnIndex = values.playerIdToColumnIndex.at(curPlayerId);
        // std::cout << output.size(0) << "," << output.size(1) << std::endl;
        for (size_t radialVelIndex = 0; radialVelIndex < csknow::weapon_speed::num_radial_bins; radialVelIndex++) {
            if ((useRealProbT && teamId == ENGINE_TEAM_T) || (useRealProbCT && teamId == ENGINE_TEAM_CT)) {
                result.radialVelProbabilities.push_back(output[0][columnIndex][0][radialVelIndex].item<float>());
            }
            else {
                result.radialVelProbabilities.push_back(1. / csknow::weapon_speed::num_radial_bins);
            }
            if (result.radialVelProbabilities.back() > mostLikelyRadialVelProb) {
                mostLikelyRadialVelProb = result.radialVelProbabilities.back();
                result.mostLikelyRadialVel = static_cast<int64_t>(radialVelIndex);
            }
        }
        return result;

    }
}
