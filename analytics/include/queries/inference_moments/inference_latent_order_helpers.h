//
// Created by durst on 4/13/23.
//

#ifndef CSKNOW_INFERENCE_LATENT_ORDER_HELPERS_H
#define CSKNOW_INFERENCE_LATENT_ORDER_HELPERS_H

#include "queries/inference_moments/inference_latent_engagement_helpers.h"

namespace csknow::inference_latent_order {
    enum class OrderRole {
        A0,
        A1,
        A2,
        B0,
        B1,
        B2
    };

    constexpr int total_orders = feature_store::num_orders_per_site * 2;

    struct InferenceOrderTickValues {
        std::vector<float> rowCPP;
        map<int64_t, size_t> playerIdToColumnIndex;
    };
    InferenceOrderTickValues extractFeatureStoreOrderValues(
        const csknow::feature_store::FeatureStoreResult & featureStoreResult, int64_t rowIndex);
    struct InferenceOrderPlayerAtTickProbabilities {
        vector<float> orderProbabilities;
        OrderRole mostLikelyOrder;
    };
    InferenceOrderPlayerAtTickProbabilities extractFeatureStoreOrderResults(
        const at::Tensor & output, const InferenceOrderTickValues & values, int64_t curPlayerId);
}

#endif //CSKNOW_INFERENCE_LATENT_ORDER_HELPERS_H
