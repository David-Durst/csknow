//
// Created by durst on 4/20/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_PLACE_HELPERS_H
#define CSKNOW_INFERENCE_LATENT_PLACE_HELPERS_H

#include "queries/inference_moments/inference_latent_engagement_helpers.h"

namespace csknow::inference_latent_place {
    struct InferencePlaceTickValues {
        std::vector<float> rowCPP;
        map<int64_t, size_t> playerIdToColumnIndex;
    };
    InferencePlaceTickValues extractFeatureStorePlaceValues(
            const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex);
    struct InferencePlacePlayerAtTickProbabilities {
        vector<float> placeProbabilities;
        PlaceIndex mostLikelyPlace;
    };
    InferencePlacePlayerAtTickProbabilities extractFeatureStorePlaceResults(
            const at::Tensor & output, const InferencePlaceTickValues & values, int64_t curPlayerId);
}

#endif //CSKNOW_INFERENCE_LATENT_PLACE_HELPERS_H
