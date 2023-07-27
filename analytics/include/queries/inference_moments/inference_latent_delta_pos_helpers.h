//
// Created by durst on 4/20/23.
//

#ifndef CSKNOW_INFERENCE_DELTA_POS_HELPERS_H
#define CSKNOW_INFERENCE_DELTA_POS_HELPERS_H

#include "queries/inference_moments/inference_latent_engagement_helpers.h"
#include "bots/behavior_tree/inference_control_parameters.h"

namespace csknow::inference_delta_pos {
    struct InferenceDeltaPosTickValues {
        std::vector<float> rowCPP;
        map<int64_t, size_t> playerIdToColumnIndex;
    };
    InferenceDeltaPosTickValues extractFeatureStoreDeltaPosValues(
            const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex,
            TeamSaveControlParameters teamSaveControlParameters);
    struct InferenceDeltaPosPlayerAtTickProbabilities {
        //vector<float> deltaPosProbabilities;
        //int64_t mostLikelyDeltaPos;
        vector<float> radialVelProbabilities;
        int64_t mostLikelyRadialVel;
    };
    InferenceDeltaPosPlayerAtTickProbabilities extractFeatureStoreDeltaPosResults(
            const at::Tensor & output, const InferenceDeltaPosTickValues & values, int64_t curPlayerId, TeamId teamId);
}

#endif //CSKNOW_INFERENCE_DELTA_POS_HELPERS_H
