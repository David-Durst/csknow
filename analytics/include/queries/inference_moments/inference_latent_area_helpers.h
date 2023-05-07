//
// Created by durst on 4/20/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_AREA_HELPERS_H
#define CSKNOW_INFERENCE_LATENT_AREA_HELPERS_H

#include "queries/inference_moments/inference_latent_engagement_helpers.h"

namespace csknow::inference_latent_area {
    struct InferenceAreaTickValues {
        std::vector<float> rowCPP;
        map<int64_t, size_t> playerIdToColumnIndex;
    };
    InferenceAreaTickValues extractFeatureStoreAreaValues(
        const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex);
    struct InferenceAreaPlayerAtTickProbabilities {
        vector<float> areaProbabilities;
        size_t mostLikelyArea;
    };
    InferenceAreaPlayerAtTickProbabilities extractFeatureStoreAreaResults(
        const at::Tensor & output, const InferenceAreaTickValues & values, int64_t curPlayerId, TeamId teamId);
}

#endif //CSKNOW_INFERENCE_LATENT_AREA_HELPERS_H
